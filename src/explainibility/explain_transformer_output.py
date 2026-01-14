import argparse
from pyaiwrap.datasets import KneeMRISegmentationDataset
from pyaiwrap.config import loadConfig
from pyaiwrap.utils import prepareDevice
from pyaiwrap.generator import createGenerator
from torch.utils.data import DataLoader
from pyaiwrap.xai import LIMEExplainer, SaliencyExplainer, GradCAMExplainer
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pathlib import Path
from datetime import datetime


def parseCMDArgs():
    parser = argparse.ArgumentParser(description="Train generator model with configurable hyperparameters.")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="./hyperparams_generator/0.json",
        help="Path to the JSON file containing configuration for training."
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index of sample from dataset to explain (default: 0)"
    )
    args = parser.parse_args()
    return args


def createDataLoader(config):
    """Create train and validation data loaders for knee MRI segmentation using train_test_split."""
    full_dataset = KneeMRISegmentationDataset(
        data_root=config["DATA_PATH"],
        metadata_path=config["DATA_PATH"] + "/metadata.csv",
        target_size=config["RESIZE"]
    )

    data_loader = DataLoader(
        full_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=2
    )

    return data_loader, len(full_dataset)


def createFeatureMask(input_tensor: torch.Tensor, block_size: int = 16):
    """Create feature mask by grouping voxels into blocks."""
    _, _, D, H, W = input_tensor.shape

    feature_mask = torch.zeros((1, 1, D, H, W), dtype=torch.long,
                               device=input_tensor.device)

    group_idx = 0
    for d in range(0, D, block_size):
        for h in range(0, H, block_size):
            for w in range(0, W, block_size):
                d_end = min(d + block_size, D)
                h_end = min(h + block_size, H)
                w_end = min(w + block_size, W)
                feature_mask[:, :, d:d_end, h:h_end, w:w_end] = group_idx
                group_idx += 1

    return feature_mask


def simpleDualGrid(input_volume, explanation, method_name="XAI", save_path=None, dpi=150, alpha=0.7):

    if input_volume.dim() == 5:
        input_np = input_volume[0, 0].cpu().numpy()
        exp_np = explanation[0, 0].cpu().numpy()
    else:
        input_np = input_volume[0].cpu().numpy()
        exp_np = explanation[0].cpu().numpy()
    
    rows, cols = 4, 8
    slices_per_image = rows * cols
    
    # 1. Grayscale to [0, 1]
    p_low, p_high = np.percentile(input_np, [5, 95])  # Use wider percentiles
    if p_high > p_low:
        mri_norm = np.clip((input_np - p_low) / (p_high - p_low), 0, 1)
    else:
        mri_norm = input_np / (input_np.max() + 1e-8)
    
    # Apply gamma to make knees more visible
    mri_norm = np.power(mri_norm, 0.6)

    if np.min(exp_np) >= 0:
        # SALIENCY - only positive (red)
        exp_max = np.percentile(exp_np, 99) if np.percentile(exp_np, 99) > 0 else exp_np.max()
        if exp_max > 0:
            exp_normalized = exp_np / exp_max
            # Boost visibility
            exp_normalized = np.power(exp_normalized, 0.5)
        else:
            exp_normalized = exp_np
        
        # Colormap: black -> red
        colors = ['black', 'red']
        cmap = mcolors.LinearSegmentedColormap.from_list('positive_only', colors)
        norm = mcolors.Normalize(vmin=0, vmax=1)
        attribution_type = "(Red=Positive Attribution)"
        
    else:
        # LIME - positive (red) and negative (blue)
        pos_values = np.maximum(exp_np, 0)
        neg_values = np.maximum(-exp_np, 0)
        
        # Scale independently
        pos_max = np.percentile(pos_values, 95) if np.percentile(pos_values, 95) > 0 else pos_values.max()
        neg_max = np.percentile(neg_values, 95) if np.percentile(neg_values, 95) > 0 else neg_values.max()
        
        if pos_max > 0:
            pos_scaled = pos_values / pos_max
            pos_scaled = np.power(pos_scaled, 0.6)
        else:
            pos_scaled = pos_values
            
        if neg_max > 0:
            neg_scaled = neg_values / neg_max
            neg_scaled = np.power(neg_scaled, 0.6)
        else:
            neg_scaled = neg_values
        
        # Combine: positive stays positive, negative stays negative
        exp_normalized = pos_scaled - neg_scaled
        
        # Colormap: blue -> black -> red (exactly as requested)
        colors = ['blue', 'black', 'red']
        cmap = mcolors.LinearSegmentedColormap.from_list('bipolar_cmap', colors)
        norm = mcolors.Normalize(vmin=-1, vmax=1)
        attribution_type = "(Blue=Negative, Red=Positive Attribution)"
    
    # Get slice dimensions
    slice_height, slice_width = input_np.shape[1], input_np.shape[2]
    grid_height = slice_height * rows
    grid_width = slice_width * cols
    grid = np.zeros((grid_height, grid_width, 3))
    
    # Fill grid with slices
    for i in range(min(slices_per_image, input_np.shape[0])):
        row = i // cols
        col = i % cols
        
        y_start = row * slice_height
        y_end = (row + 1) * slice_height
        x_start = col * slice_width
        x_end = (col + 1) * slice_width
        
        mri_slice = mri_norm[i]
        exp_slice = exp_normalized[i]

        if np.min(exp_np) >= 0:
            # SALIENCY: Only red positive attributions
            
            # Black background
            red_channel = np.zeros_like(mri_slice)
            green_channel = np.zeros_like(mri_slice)
            blue_channel = np.zeros_like(mri_slice)
            
            # Gray knees (MRI) - all channels equal
            gray_intensity = mri_slice * 0.6  # Control gray darkness
            red_channel += gray_intensity
            green_channel += gray_intensity
            blue_channel += gray_intensity
            
            # Red positive attributions
            # Red channel gets attribution + MRI
            red_attribution = exp_slice * 1.2  # Boost red attribution
            red_channel += red_attribution

            red_channel = np.clip(red_channel, 0, 1)
            green_channel = np.clip(green_channel, 0, 1)
            blue_channel = np.clip(blue_channel, 0, 1)
            
        else:
            # LIME: Blue negative, Red positive
            
            # Black background
            red_channel = np.zeros_like(mri_slice)
            green_channel = np.zeros_like(mri_slice)
            blue_channel = np.zeros_like(mri_slice)
            
            # Gray knees (MRI)
            gray_intensity = mri_slice * 0.5  # Slightly darker to make colors pop
            red_channel += gray_intensity
            green_channel += gray_intensity
            blue_channel += gray_intensity
            
            # Extract positive (red) and negative (blue) components
            pos_mask = exp_slice > 0
            neg_mask = exp_slice < 0
            
            # Red positive attributions
            if np.any(pos_mask):
                pos_strength = exp_slice[pos_mask]
                # Scale and boost
                pos_scaled = pos_strength * 1.5
                red_channel[pos_mask] += pos_scaled
                # Keep some gray in positive areas
                green_channel[pos_mask] += gray_intensity[pos_mask] * 0.3
                blue_channel[pos_mask] += gray_intensity[pos_mask] * 0.3
            
            #  Blue negative attributions  
            if np.any(neg_mask):
                neg_strength = -exp_slice[neg_mask]  # Convert to positive
                # Scale and boost
                neg_scaled = neg_strength * 1.5
                blue_channel[neg_mask] += neg_scaled
                # Keep some gray in negative areas
                red_channel[neg_mask] += gray_intensity[neg_mask] * 0.3
                green_channel[neg_mask] += gray_intensity[neg_mask] * 0.3
            
            # Clip to valid range
            red_channel = np.clip(red_channel, 0, 1)
            green_channel = np.clip(green_channel, 0, 1)
            blue_channel = np.clip(blue_channel, 0, 1)
        
        # Combine channels
        rgb_slice = np.stack([red_channel, green_channel, blue_channel], axis=-1)
        grid[y_start:y_end, x_start:x_end, :] = rgb_slice

    fig = plt.figure(figsize=(24, 12))
    ax = plt.gca()
    
    # Display the grid
    ax.imshow(grid)

    title_text = f'MRI Slices with {method_name} Explanations\n{attribution_type}'
    ax.set_title(title_text, fontsize=16, pad=20, fontweight='bold', color='white')
    title = ax.title
    title.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8, edgecolor='white'))
    
    ax.axis('off')
    
    # Slice numbers
    for i in range(min(slices_per_image, input_np.shape[0])):
        row = i // cols
        col = i % cols
        text_x = col * slice_width + 5
        text_y = row * slice_height + 15
        
        ax.text(text_x, text_y, f'S{i+1:02d}', 
                fontsize=9, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8, edgecolor='white'))
    
    # Legend
    if np.min(exp_np) >= 0:
        legend_elements = [
            Patch(facecolor='red', alpha=0.9, edgecolor='white', linewidth=2, label='Positive Attribution'),
            Patch(facecolor='gray', alpha=0.7, edgecolor='white', linewidth=1, label='Knee MRI (Gray)'),
            Patch(facecolor='black', alpha=1.0, edgecolor='white', linewidth=1, label='Background (Black)'),
        ]
    else:
        legend_elements = [
            Patch(facecolor='red', alpha=0.9, edgecolor='white', linewidth=2, label='Positive Attribution'),
            Patch(facecolor='blue', alpha=0.9, edgecolor='white', linewidth=2, label='Negative Attribution'),
            Patch(facecolor='gray', alpha=0.7, edgecolor='white', linewidth=1, label='Knee MRI (Gray)'),
            Patch(facecolor='black', alpha=1.0, edgecolor='white', linewidth=1, label='Background (Black)'),
        ]
    
    legend = ax.legend(handles=legend_elements,
                       loc='upper right',
                       bbox_to_anchor=(0.98, 0.98),
                       fontsize=10,
                       framealpha=0.95,
                       frameon=True,
                       labelcolor='white')
    legend.get_frame().set_facecolor('black')
    legend.get_frame().set_edgecolor('white')
    legend.get_frame().set_linewidth(2)
    
    # Colorbar
    fig.subplots_adjust(bottom=0.15)
    cax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
    cax.set_facecolor('black')
    
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=cax, orientation='horizontal')
    
    if np.min(exp_np) >= 0:
        cbar_label = f'{method_name} Attribution Intensity (Red)'
    else:
        cbar_label = 'Attribution Intensity (Blue=Negative, Red=Positive)'
    
    cbar.set_label(cbar_label, fontsize=12, labelpad=10, color='white')
    cbar.ax.tick_params(labelsize=10, colors='white')
    cax.xaxis.label.set_color('white')
    cax.tick_params(axis='x', colors='white')
    
    # Grid lines
    for i in range(1, rows):
        ax.axhline(y=i * slice_height, color='white', linewidth=0.7, alpha=0.3, linestyle='--')
    for i in range(1, cols):
        ax.axvline(x=i * slice_width, color='white', linewidth=0.7, alpha=0.3, linestyle='--')
    
    # Dark background
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = save_path.parent / f"{save_path.stem}_{timestamp}{save_path.suffix}"
        
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                    facecolor='black', edgecolor='none',
                    transparent=False)
    
    return fig, save_path if save_path else None


def saveRawData(voxels, explanation, sample_idx, output_dir):
    """Save raw tensors for further analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sample_{sample_idx}_{timestamp}_raw_data.pth"
    filepath = output_dir / filename

    torch.save({
        'input': voxels.cpu(),
        'explanation': explanation.cpu(),
        'sample_idx': sample_idx,
        'timestamp': timestamp
    }, filepath)

    return filepath


def getSampleFromDataloader(data_loader, sample_idx):
    """Get a specific sample from dataloader by index."""
    # Reset dataloader to ensure consistent indexing
    data_loader_iter = iter(data_loader)

    for i in range(sample_idx + 1):
        try:
            voxels, mask = next(data_loader_iter)
        except StopIteration:
            raise IndexError(f"Sample index {sample_idx} out of range. Dataset has fewer samples.")

    return voxels, mask


def main():
    args = parseCMDArgs()
    config = loadConfig(args.config)
    config["BATCH_SIZE"] = 1  # For LIME, use batch size of 1
    device = prepareDevice()
    data_loader, dataset_size = createDataLoader(config)
    generator = createGenerator(config=config, device=device)
    generator.load_state_dict(torch.load("../../weights/segmentator_segformer_hyperparams_SegFormer3D.pth", map_location=device))
    generator = generator.eval()
    lime_explainer = LIMEExplainer(
        n_samples=100,           # Number of LIME samples
        batch_size=1,           # Batch size for processing
        segmentation_mode=True,  # Vital for 3D segmentation
    )

    print(f"Dataset size: {dataset_size} samples")

    if args.sample_idx >= dataset_size:
        print(f"Error: Sample index {args.sample_idx} out of range (0-{dataset_size-1})")
        return

    print(f"\nLoading sample {args.sample_idx}...")
    voxels, _ = getSampleFromDataloader(data_loader, args.sample_idx)
    voxels = voxels.to(device)
    feature_mask = createFeatureMask(voxels, block_size=6)
    feature_mask = feature_mask.to(device)
    mask = feature_mask
    # mask = mask.to(device)

    explanation = lime_explainer.explain(
        model=generator,
        input_tensor=voxels,
        class_idx=None,
        show_progress=True,
        feature_mask=mask,
        return_input_shape=True,   # Return same shape as input
    )
    print("\nExplanation Statistics (LIME):")
    print(f"  Shape: {explanation.shape}")
    print(f"  Range: [{explanation.min():.6f}, {explanation.max():.6f}]")
    print(f"  Mean: {explanation.mean():.6f}")
    print(f"  Std: {explanation.std():.6f}")
    simpleDualGrid(voxels, explanation, "LIME", save_path=f"../../diagrams/SegFormer3D_kneemridataset_lime_explanation_{args.sample_idx}.png", dpi=250, alpha=1.0)

    saveRawData(voxels, explanation, args.sample_idx, output_dir="../../LIME/")
    explainer = SaliencyExplainer(absolute=True)

    explanation = explainer.explain(
        model=generator,
        input_tensor=voxels,
        target_class=None,
    )

    print("\nExplanation Statistics (Saliency):")
    print(f"  Shape: {explanation.shape}")
    print(f"  Range: [{explanation.min():.6f}, {explanation.max():.6f}]")
    print(f"  Mean: {explanation.mean():.6f}")
    print(f"  Std: {explanation.std():.6f}")
    simpleDualGrid(voxels, explanation, "Saliency", save_path=f"../../diagrams/SegFormer3D_kneemridataset_saliency_explanation_{args.sample_idx}.png", dpi=250, alpha=1.0)

    saveRawData(voxels, explanation, args.sample_idx, output_dir="../../Saliency/")
    explainer = GradCAMExplainer()

    explanation = explainer.explain(
        model=generator,
        input_tensor=voxels,
        target_class=None
    )

    print("\nExplanation Statistics (GradCAM):")
    print(f"  Shape: {explanation.shape}")
    print(f"  Range: [{explanation.min():.6f}, {explanation.max():.6f}]")
    print(f"  Mean: {explanation.mean():.6f}")
    print(f"  Std: {explanation.std():.6f}")
    simpleDualGrid(voxels, explanation, "GradCAM", save_path=f"../../diagrams/SegFormer3D_kneemridataset_gradcam_explanation_{args.sample_idx}.png", dpi=250, alpha=1.0)

    saveRawData(voxels, explanation, args.sample_idx, output_dir="../../GradCAM/")


if __name__ == "__main__":
    main()
