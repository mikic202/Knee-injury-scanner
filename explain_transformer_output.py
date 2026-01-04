import argparse
from pyaiwrap.datasets import KneeMRISegmentationDataset
from pyaiwrap.config import loadConfig
from pyaiwrap.utils import prepareDevice
from pyaiwrap.generator import createGenerator
from torch.utils.data import DataLoader
from pyaiwrap.xai import LIMEExplainer
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


def simpleDualGrid(input_volume, explanation, save_path=None, dpi=150, alpha=0.4):
    """
    Combined visualization of input MRI with overlaid explanations in a single grid.

    Args:
        input_volume: Input tensor [B, C, D, H, W] or [B, D, H, W]
        explanation: Explanation tensor [B, C, D, H, W] or [B, D, H, W]
        save_path: Optional path to save the figure (str or Path)
        dpi: Resolution for saved image
        alpha: Transparency for overlay (0-1)
    """
    if input_volume.dim() == 5:
        input_np = input_volume[0, 0].cpu().numpy()
        exp_np = explanation[0, 0].cpu().numpy()
    else:
        input_np = input_volume[0].cpu().numpy()
        exp_np = explanation[0].cpu().numpy()

    rows, cols = 4, 8  # 4x8 = 32 slices
    slices_per_image = rows * cols

    # Create figure with proper sizing for readability
    fig = plt.figure(figsize=(24, 12))

    # Create main axes for the grid
    ax = plt.gca()

    # Create normalization for explanations
    exp_max = np.abs(exp_np).max()
    exp_normalized = np.clip(exp_np, -exp_max, exp_max) / exp_max if exp_max > 0 else exp_np

    # Get slice dimensions
    slice_height, slice_width = input_np.shape[1], input_np.shape[2]

    # Create empty RGB grid
    grid_height = slice_height * rows
    grid_width = slice_width * cols
    grid = np.zeros((grid_height, grid_width, 3))

    # Fill grid with slices
    for i in range(min(slices_per_image, input_np.shape[0])):
        row = i // cols
        col = i % cols

        # Calculate grid position
        y_start = row * slice_height
        y_end = (row + 1) * slice_height
        x_start = col * slice_width
        x_end = (col + 1) * slice_width

        # Get current slice
        input_slice = input_np[i]
        exp_slice = exp_normalized[i]

        # Normalize input for grayscale display
        input_min, input_max = input_slice.min(), input_slice.max()
        if input_max > input_min:
            input_normalized = (input_slice - input_min) / (input_max - input_min)
        else:
            input_normalized = input_slice

        # Create RGB representation
        # Red channel: positive explanations
        red = np.zeros_like(input_normalized)
        pos_mask = exp_slice > 0
        red[pos_mask] = exp_slice[pos_mask]

        # Blue channel: negative explanations
        blue = np.zeros_like(input_normalized)
        neg_mask = exp_slice < 0
        blue[neg_mask] = -exp_slice[neg_mask]

        # Green channel: grayscale background (input)
        green = input_normalized.copy()

        rgb_slice = np.stack([red, green, blue], axis=-1)

        # Apply alpha blending
        rgb_slice = input_normalized[:, :, None] * (1 - alpha) + rgb_slice * alpha

        # Place in grid
        grid[y_start:y_end, x_start:x_end, :] = rgb_slice

    # Display the grid
    im = ax.imshow(grid, vmin=0, vmax=1)

    # Title with white text for better visibility
    ax.set_title('MRI Slices with LIME Explanations Overlay\n(Red=Positive Attribution, Blue=Negative Attribution)', 
                 fontsize=16, pad=20, fontweight='bold', color='white')

    # Add a subtle background to the title for better contrast
    title = ax.title
    title.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='white'))

    ax.axis('off')

    # Add slice numbers to each grid cell for better orientation
    for i in range(min(slices_per_image, input_np.shape[0])):
        row = i // cols
        col = i % cols

        # Position for text (top-left corner of each slice)
        text_x = col * slice_width + 5
        text_y = row * slice_height + 15

        ax.text(text_x, text_y, f'S{i+1:02d}', 
                fontsize=9, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8, edgecolor='white'))

    # Legend with white text
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
                       labelcolor='white')  # Make legend text white

    legend.get_frame().set_facecolor('black')
    legend.get_frame().set_edgecolor('white')
    legend.get_frame().set_linewidth(2)

    # Add comprehensive statistics section with black text
    stats_text = f"""
    Explanation Statistics:
    • Range: [{exp_np.min():.4f}, {exp_np.max():.4f}]
    • Mean Absolute: {np.abs(exp_np).mean():.4f}
    • Std Dev: {exp_np.std():.4f}
    • Positive %: {(exp_np > 0).sum() / exp_np.size:.1%}
    • Negative %: {(exp_np < 0).sum() / exp_np.size:.1%}

    Input Statistics:
    • Range: [{input_np.min():.2f}, {input_np.max():.2f}]
    • Mean: {input_np.mean():.2f}
    • Slices: {input_np.shape[0]} total, {min(slices_per_image, input_np.shape[0])} shown
    """

    # Create text box for statistics with white text on black background
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            color='black',  # Black text for contrast
            verticalalignment='top',
            bbox=props,
            fontfamily='monospace')

    # Add color intensity bar with white labels
    fig.subplots_adjust(bottom=0.15)
    cax = fig.add_axes([0.25, 0.05, 0.5, 0.02])  # [left, bottom, width, height]

    # Set dark background for colorbar
    cax.set_facecolor('black')

    colors = ['blue', 'white', 'red']
    cmap = mcolors.LinearSegmentedColormap.from_list('attribution', colors)

    norm = mcolors.Normalize(vmin=-exp_max, vmax=exp_max)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=cax, orientation='horizontal')

    # White text for colorbar
    cbar.set_label('Attribution Intensity (Blue=Negative, Red=Positive)', 
                   fontsize=12, labelpad=10, color='white')
    cbar.ax.tick_params(labelsize=10, colors='white')

    # Set colorbar axis colors to white
    cax.xaxis.label.set_color('white')
    cax.tick_params(axis='x', colors='white')

    # Add grid lines for better slice separation
    for i in range(1, rows):
        ax.axhline(y=i * slice_height, color='white', linewidth=0.7, alpha=0.4, linestyle='--')
    for i in range(1, cols):
        ax.axvline(x=i * slice_width, color='white', linewidth=0.7, alpha=0.4, linestyle='--')

    # Set figure background to black for better contrast with the grid
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.exists():
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = save_path.parent / f"{save_path.stem}_{timestamp}{save_path.suffix}"

        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                    facecolor='black', edgecolor='none',  # Black background for saved image
                    transparent=False)
        print(f"Figure saved to: {save_path}")

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
    generator.load_state_dict(torch.load("./weights/segmentator_segformer_hyperparams_SegFormer3D.pth", map_location=device))
    generator = generator.eval()
    lime_explainer = LIMEExplainer(
        n_samples=50,           # Number of LIME samples
        batch_size=1,           # Batch size for processing
        segmentation_mode=True,  # Vital for 3D segmentation
    )

    print(f"Dataset size: {dataset_size} samples")

    if args.sample_idx >= dataset_size:
        print(f"Error: Sample index {args.sample_idx} out of range (0-{dataset_size-1})")
        return

    print(f"\nLoading sample {args.sample_idx}...")
    voxels, mask = getSampleFromDataloader(data_loader, args.sample_idx)
    voxels = voxels.to(device)
    feature_mask = createFeatureMask(voxels, block_size=16)
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
    print(f"Explanation shape: {explanation.shape}")
    print(f"Explanation range: [{explanation.min():.6f}, {explanation.max():.6f}]")
    simpleDualGrid(voxels, explanation, save_path=f"./diagrams/SegFormer3D_kneemridataset_lime_explanation_{args.sample_idx}.png", dpi=250, alpha=1.0)


if __name__ == "__main__":
    main()
