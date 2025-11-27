import argparse
from pyaiwrap.train import train
from pyaiwrap.datasets import KneeMRISegmentationDataset
from pyaiwrap.loss import SegmentationDiceCELoss
from pyaiwrap.metrics import SegmentationMetrics
from pyaiwrap.config import loadConfig
from pyaiwrap.optimizers import createOptimizer
from pyaiwrap.schedulers import createScheduler
from pyaiwrap.utils import prepareDevice
from pyaiwrap.generator import createGenerator
from pyaiwrap.control import SegmentationControlFunc
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


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
        "--launch_number",
        type=int,
        required=False,
        default=0,
        help="The number of the training process launch with the same hyperparams file (increase it for subsequent runs\
 with the same hyperparams file)."
    )
    args = parser.parse_args()
    return args


def createDataLoaders(config):
    """Create train and validation data loaders for knee MRI segmentation using train_test_split."""
    full_dataset = KneeMRISegmentationDataset(
        data_root=config["DATA_PATH"],
        metadata_path=config["DATA_PATH"] + "/metadata.csv",
        target_size=config["RESIZE"]
    )

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.1,
        random_state=42,
        shuffle=True,
        stratify=full_dataset.metadata.iloc[full_dataset.valid_samples]['aclDiagnosis']
    )

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    validation_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=8
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=8
    )

    return train_loader, validation_loader, len(train_dataset), len(validation_dataset)


def createSegmentator(config, device):
    """Create and return the generator model."""
    generator = createGenerator(config=config, device=device)
    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)

    return generator, total_params, trainable_params


def prepareOptimizer(model_parameters, config):
    """Create optimizer."""
    optimizer = createOptimizer(model_parameters, config)
    return optimizer


def prepareScheduler(optimizer, config, train_loader_len):
    """Create scheduler."""
    scheduler = createScheduler(
        optimizer=optimizer,
        config=config,
        train_loader_len=train_loader_len
    )
    return scheduler


def createLossFunction(config, device):
    """Create the loss function."""
    loss_fn = SegmentationDiceCELoss(
        dice_weight=config["DICE_WEIGHT"],
        ce_weight=config["CE_WEIGHT"]
    )
    return loss_fn


def createMetrics():
    """Create metrics object."""
    metrics = SegmentationMetrics()
    return metrics


def main():
    args = parseCMDArgs()
    device = prepareDevice()
    config = loadConfig(args.config)

    train_loader, validation_loader, train_samples, val_samples = createDataLoaders(config)
    print(f"Training samples: {train_samples}")
    print(f"Validation samples: {val_samples}\n")

    print("Building segmentator model...")
    segmentator, total_params, trainable_params = createSegmentator(config, device)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    optimizer = prepareOptimizer(segmentator.parameters(), config)
    optimizers = {'segformer': optimizer}

    scheduler = prepareScheduler(optimizer, config, len(train_loader))
    schedulers = {'segformer': scheduler}

    models = {'segformer': segmentator}

    loss_fn = createLossFunction(config, device)
    metrics = createMetrics()

    train(
        models=models,
        optimizers=optimizers,
        schedulers=schedulers,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_fn=loss_fn,
        metrics=metrics,
        device=device,
        control_fn=SegmentationControlFunc(num_classes=4),
        launch_number=args.launch_number,
        num_epochs=config["EPOCHS"],
        diagrams_data_path=config["DIAGRAMS_DATA_PATH"],
        hyperparams_id=config["HYPERPARAMS_ID"],
        weights_path=config["WEIGHTS_PATH"],
        diagrams_path=config["DIAGRAMS_PATH"],
        visualize_every_xth_epoch=config["VISUALIZE_EVERY"],
        max_patience=config["PATIENCE"],
        model_type="segmentator",
        gradient_clip=config["GRADIENT_CLIP"],
        early_stopping_metric="total_loss"
    )


if __name__ == "__main__":
    main()
