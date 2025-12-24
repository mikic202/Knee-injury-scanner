import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, Normalize
import math
import torch.nn as nn
import torch
from src.explainibility.basic_gradient_based_methods import (
    explain_prediction_with_deconvolution,
    explain_prediction_with_guided_backprop,
    explain_prediction_with_input_x_gradient,
    explain_prediction_with_saliency,
    explain_prediction_with_integrated_gradients,
)
from pathlib import Path


def display_explainibility(
    example_values: np.ndarray,
    attributions_values: np.ndarray,
    example_minimal_value: float = 100.0,
    atributions_minimal_value: float = 0.0001,
    example_alpha_visibility: float = 0.1,
    figsize: tuple[int, int] = (12, 12),
    z_slize_size: int = 10,
    display: bool = True,
    title: str = "3D Explainibility Visualization",
) -> None:
    example_mask = example_values >= example_minimal_value
    example = example_values * example_mask

    attributions_mask = attributions_values >= atributions_minimal_value
    attributions = attributions_values * attributions_mask

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    S, H, W = example.shape
    X_coords = np.arange(W)
    Y_coords = np.arange(H)
    X_mesh, Y_mesh = np.meshgrid(X_coords, Y_coords)

    gray_cmap = cm.get_cmap("gray")
    gray_transparent_cmap = gray_cmap(np.arange(gray_cmap.N))
    gray_transparent_cmap[0, 3] = 0.0
    transparent_cmap = ListedColormap(gray_transparent_cmap)

    red_cmap = cm.get_cmap("Reds")
    red_transparent_cmap = red_cmap(np.arange(red_cmap.N))
    red_transparent_cmap[0, 3] = 0.0
    attributions_cmap = ListedColormap(red_transparent_cmap)

    example_norm = Normalize(vmin=0, vmax=example.max())
    attributions_norm = Normalize(vmin=attributions.min(), vmax=attributions.max())

    for i in range(S):
        Z_plane = i * z_slize_size * np.ones_like(X_mesh)
        example_slice_data = example[i, :, :]
        attributions_slice_data = attributions[i, :, :]

        example_rgba_colors = transparent_cmap(example_norm(example_slice_data))
        attributions_rgba_colors = attributions_cmap(
            attributions_norm(attributions_slice_data)
        )

        ax.plot_surface(
            X_mesh,
            Y_mesh,
            Z_plane,
            facecolors=example_rgba_colors,
            rcount=50,
            ccount=50,
            shade=False,
            alpha=example_alpha_visibility,
            linewidth=0.0,
        )
        ax.plot_surface(
            X_mesh,
            Y_mesh,
            Z_plane + 0.5,
            facecolors=attributions_rgba_colors,
            rcount=50,
            ccount=50,
            shade=False,
        )

    plt.title(title)

    if display:
        plt.tight_layout()
        plt.show()


def display_explainibility_in_slices(
    example_values: np.ndarray,
    attributions_values: np.ndarray,
    example_minimal_value: float = 100.0,
    attributions_minimal_value: float = 0.0001,
    figsize: tuple[int, int] = (12, 12),
    display: bool = True,
    title: str = "Explainibility Slices Visualization",
):
    S = example_values.shape[0]

    example = np.where(example_values >= example_minimal_value, example_values, np.nan)
    attributions = np.where(
        attributions_values >= attributions_minimal_value, attributions_values, np.nan
    )

    cols = int(math.ceil(np.sqrt(S)))
    rows = int(math.ceil(S / cols))
    _, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    example_norm = Normalize(vmin=np.nanmin(example), vmax=np.nanmax(example))
    attributions_norm = Normalize(
        vmin=np.nanmin(attributions), vmax=np.nanmax(attributions)
    )

    for i in range(S):
        ax = axes[i]
        ax.set_title(f"Slice {i}")

        ax.imshow(example[i], cmap="gray", norm=example_norm)
        ax.imshow(attributions[i], cmap="Reds", norm=attributions_norm, alpha=0.6)

        ax.axis("off")

    for j in range(S, len(axes)):
        axes[j].axis("off")

    plt.title(title)

    if display:
        plt.tight_layout()
        plt.show()


def get_ideal_minimal_atribution_value(
    attributions_values: np.ndarray,
):
    return attributions_values.max() * 0.2


def dislay_all_explainibility(
    model: nn.Module,
    example: np.ndarray,
    target_class: int,
    device: torch.device | None = None,
    display: bool = True,
    save_path: Path | None = None,
) -> None:
    if not save_path:
        save_path = Path("")

    silency_atributions = explain_prediction_with_saliency(
        model, example, target_class, device
    )

    display_explainibility(
        example.squeeze(0),
        silency_atributions.squeeze(0),
        atributions_minimal_value=get_ideal_minimal_atribution_value(
            silency_atributions
        ),
        display=False,
        title="Saliency Explainibility Visualization",
    )

    plt.savefig(save_path / "saliency_explainibility.png")

    deconvolution_atribution = explain_prediction_with_deconvolution(
        model, example, target_class, device
    )

    display_explainibility(
        example.squeeze(0),
        deconvolution_atribution.squeeze(0),
        atributions_minimal_value=get_ideal_minimal_atribution_value(
            deconvolution_atribution
        ),
        display=False,
        title="Deconvolution Explainibility Visualization",
    )

    plt.savefig(save_path / "deconvolution_explainibility.png")

    guided_backprop_atribution = explain_prediction_with_guided_backprop(
        model, example, target_class, device
    )

    display_explainibility(
        example.squeeze(0),
        guided_backprop_atribution.squeeze(0),
        atributions_minimal_value=get_ideal_minimal_atribution_value(
            guided_backprop_atribution
        ),
        display=False,
        title="Guided Backpropagation Explainibility Visualization",
    )

    plt.savefig(save_path / "guided_backprop_explainibility.png")

    input_x_gradient_atribution = explain_prediction_with_input_x_gradient(
        model, example, target_class, device
    )

    display_explainibility(
        example.squeeze(0),
        input_x_gradient_atribution.squeeze(0),
        atributions_minimal_value=get_ideal_minimal_atribution_value(
            input_x_gradient_atribution
        ),
        display=False,
        title="Input x Gradient Explainibility Visualization",
    )

    plt.savefig(save_path / "input_x_gradient_explainibility.png")

    integrated_gradients_atribution = explain_prediction_with_integrated_gradients(
        model, example, target_class, device
    )
    display_explainibility(
        example.squeeze(0),
        integrated_gradients_atribution.squeeze(0),
        atributions_minimal_value=get_ideal_minimal_atribution_value(
            integrated_gradients_atribution
        ),
        display=False,
        title="Integrated Gradients Explainibility Visualization",
    )

    plt.savefig(save_path / "integrated_gradients_explainibility.png")

    if display:
        plt.tight_layout()
        plt.show()


def display_sae_features(
    sae_statistics_per_class: dict[str, list[np.ndarray]],
    output_path: Path,
    display: bool = True,
) -> None:
    _, axes = plt.subplots(len(sae_statistics_per_class), figsize=(12, 12))
    for i, class_label in enumerate(sae_statistics_per_class):
        features = np.array(sae_statistics_per_class[class_label]).squeeze()
        ax = axes[i]
        ax.set_title(f"Features for class {class_label}")

        ax.pcolormesh(features)

    plt.savefig(output_path / "sae_features_per_class.png")

    if display:
        plt.tight_layout()
        plt.show()
