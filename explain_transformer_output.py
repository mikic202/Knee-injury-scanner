import argparse
from pyaiwrap.datasets import KneeMRISegmentationDataset
from pyaiwrap.config import loadConfig
from pyaiwrap.utils import prepareDevice
from pyaiwrap.generator import createGenerator
from torch.utils.data import DataLoader
from pyaiwrap.xai import LIMEExplainer
import torch


def parseCMDArgs():
    parser = argparse.ArgumentParser(description="Train generator model with configurable hyperparameters.")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="./hyperparams_generator/0.json",
        help="Path to the JSON file containing configuration for training."
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


def main():
    args = parseCMDArgs()
    config = loadConfig(args.config)
    config["BATCH_SIZE"] = 1  # For LIME, use batch size of 1
    device = prepareDevice()
    data_loader, dataset_size = createDataLoader(config)
    generator = createGenerator(config=config, device=device)
    generator.load_state_dict(torch.load("./weights/segmentator_segformer_hyperparams_SegFormer3D.pth", map_location=device))
    generator = generator.eval()
    # Create LIME explainer
    lime_explainer = LIMEExplainer(
        n_samples=50,           # Number of LIME samples
        batch_size=1,           # Batch size for processing
        segmentation_mode=True,  # Important for 3D segmentation
        kernel_width=0.25       # Kernel width for similarity
    )

    voxels, mask = next(iter(data_loader))
    voxels = voxels.to(device)
    # feature_mask = createFeatureMask(voxels, block_size=16)
    # feature_mask = feature_mask.to(device)
    # mask = feature_mask
    mask = mask.to(device)

    explanation = lime_explainer.explain(
        model=generator,
        input_tensor=voxels,
        show_progress=True,
        feature_mask=mask,
        return_input_shape=True,   # Return same shape as input
    )
    print(f"Explanation shape: {explanation.shape}")
    print(f"Explanation range: [{explanation.min():.6f}, {explanation.max():.6f}]")


if __name__ == "__main__":
    main()
