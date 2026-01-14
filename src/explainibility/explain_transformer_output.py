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


def simpleDualGrid(input_volume, explanation, method_name="XAI", save_path=None, dpi=150, alpha=0.4):
    """
    Combined visualization of input MRI with overlaid explanations.

    Args:
        input_volume: Input tensor [B, C, D, H, W] or [B, D, H, W]
        explanation: Explanation tensor [B, C, D, H, W] or [B, D, H, W]
        method_name: Name of XAI method for title
        save_path: Optional path to save figure
        dpi: Resolution for saved image
        alpha: Transparency for overlay (0-1)
    """
    if input_volume.dim() == 5:
        input_np = input_volume[0, 0].cpu().numpy()
        exp_np = explanation[0, 0].cpu().numpy()
    else:
        input_np = input_volume[0].cpu().numpy()
        exp_np = explanation[0].cpu().numpy()

    rows, cols = 4, 8
    slices_per_image = rows * cols

    # Create figure
    fig = plt.figure(figsize=(24, 12))
    ax = plt.gca()

    # For saliency (only positive values), use single-sided colormap
    if np.min(exp_np) >= 0:
        # ENHANCED: Use percentile normalization for better visibility of low values
        exp_max = np.percentile(exp_np, 99) if np.percentile(exp_np, 99) > 0 else exp_np.max()
        exp_normalized = exp_np / (exp_max + 1e-8)
        # ENHANCED: Apply gamma correction to boost lower values
        exp_normalized = np.power(exp_normalized, 0.5)

        colors = ['black', 'red']  # Black to red for positive-only
        cmap = mcolors.LinearSegmentedColormap.from_list('positive_attribution', colors)
        norm = mcolors.Normalize(vmin=0, vmax=1)
        attribution_type = "(Red=Positive Attribution)"
    else:
        # For LIME (positive and negative), use dual-sided colormap
        # ENHANCED: Use 99th percentile for normalization to avoid outlier dominance
        exp_abs_max = np.percentile(np.abs(exp_np), 99)
        exp_normalized = np.clip(exp_np / (exp_abs_max + 1e-8), -1, 1)

        colors = ['blue', 'white', 'red']
        cmap = mcolors.LinearSegmentedColormap.from_list('attribution', colors)
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

        input_slice = input_np[i]
        exp_slice = exp_normalized[i]

        # Normalize input
        input_min, input_max = input_slice.min(), input_slice.max()
        if input_max > input_min:
            input_normalized = (input_slice - input_min) / (input_max - input_min)
        else:
            input_normalized = input_slice

        # Create RGB representation
        if np.min(exp_np) >= 0:
            # Saliency: red channel for attribution, green for background
            # ENHANCED: Boost red channel intensity
            red = exp_slice * 2.0  # Increased from 1.0 to 2.0
            green = input_normalized * 0.7  # Reduced to make red stand out more
            blue = np.zeros_like(input_normalized)
        else:
            # LIME: red for positive, blue for negative, green for background
            # ENHANCED: Boost color intensity
            red = np.maximum(exp_slice, 0) * 1.5  # Increased from 1.0 to 1.5
            blue = np.maximum(-exp_slice, 0) * 1.5  # Increased from 1.0 to 1.5
            green = input_normalized * 0.8  # Slightly reduced

        rgb_slice = np.stack([red, green, blue], axis=-1)
        # ENHANCED: Use adaptive alpha - stronger attributions get higher alpha
        if np.min(exp_np) >= 0:
            # For saliency, use intensity-based alpha
            attr_strength = exp_slice
            adaptive_alpha = alpha * (0.3 + 0.7 * attr_strength)
        else:
            # For LIME, use absolute value for alpha
            attr_strength = np.abs(exp_slice)
            adaptive_alpha = alpha * (0.3 + 0.7 * attr_strength)

        rgb_slice = input_normalized[:, :, None] * (1 - adaptive_alpha[:, :, None]) + rgb_slice * adaptive_alpha[:, :, None]
        rgb_slice = np.clip(rgb_slice, 0, 1)  # Ensure values stay in valid range
        grid[y_start:y_end, x_start:x_end, :] = rgb_slice

    # Set title
    title_text = f'MRI Slices with {method_name} Explanations Overlay\n{attribution_type}'
    ax.set_title(title_text, fontsize=16, pad=20, fontweight='bold', color='white')
    title = ax.title
    title.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='white'))

    ax.axis('off')

    # Add slice numbers
    for i in range(min(slices_per_image, input_np.shape[0])):
        row = i // cols
        col = i % cols
        text_x = col * slice_width + 5
        text_y = row * slice_height + 15

        ax.text(text_x, text_y, f'S{i+1:02d}', 
                fontsize=9, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8, edgecolor='white'))

    # Create legend
    if np.min(exp_np) >= 0:
        legend_elements = [
            Patch(facecolor='red', alpha=0.8, edgecolor='white', linewidth=1, label='Positive Attribution'),
            Patch(facecolor='white', alpha=0.8, edgecolor='black', linewidth=1, label='Original MRI'),
        ]
    else:
        legend_elements = [
            Patch(facecolor='red', alpha=0.8, edgecolor='white', linewidth=1, label='Positive Attribution'),
            Patch(facecolor='blue', alpha=0.8, edgecolor='white', linewidth=1, label='Negative Attribution'),
            Patch(facecolor='white', alpha=0.8, edgecolor='black', linewidth=1, label='Original MRI'),
        ]

    legend = ax.legend(handles=legend_elements,
                       loc='upper right',
                       bbox_to_anchor=(0.98, 0.98),
                       fontsize=11,
                       framealpha=0.95,
                       frameon=True,
                       labelcolor='white')
    legend.get_frame().set_facecolor('black')
    legend.get_frame().set_edgecolor('white')
    legend.get_frame().set_linewidth(2)

    # Add colorbar
    fig.subplots_adjust(bottom=0.15)
    cax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
    cax.set_facecolor('black')

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=cax, orientation='horizontal')

    if np.min(exp_np) >= 0:
        cbar_label = f'{method_name} Attribution Intensity'
    else:
        cbar_label = 'Attribution Intensity (Blue=Negative, Red=Positive)'

    cbar.set_label(cbar_label, fontsize=12, labelpad=10, color='white')
    cbar.ax.tick_params(labelsize=10, colors='white')
    cax.xaxis.label.set_color('white')
    cax.tick_params(axis='x', colors='white')

    # Add grid lines
    for i in range(1, rows):
        ax.axhline(y=i * slice_height, color='white', linewidth=0.7, alpha=0.4, linestyle='--')
    for i in range(1, cols):
        ax.axvline(x=i * slice_width, color='white', linewidth=0.7, alpha=0.4, linestyle='--')

    # Set dark background
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Save figure
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
