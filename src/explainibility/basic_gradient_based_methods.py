from captum.attr import (
    Saliency,
    Deconvolution,
    GuidedBackprop,
    InputXGradient,
    IntegratedGradients,
)
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, Normalize


def explain_prediction_with_saliency(
    model: torch.nn.Module,
    example: np.ndarray,
    target_class: int,
    device: torch.device | None = None,
) -> np.ndarray:
    model.eval()
    saliency = Saliency(model)
    input_tensor = torch.tensor(example, dtype=torch.float32).unsqueeze(0)
    if device is not None:
        input_tensor = input_tensor.to(device)
    attributions = saliency.attribute(input_tensor, target=target_class)
    return attributions.squeeze(0).detach().cpu().numpy()


def explain_prediction_with_deconvolution(
    model: torch.nn.Module,
    example: np.ndarray,
    target_class: int,
    device: torch.device | None = None,
) -> np.ndarray:
    model.eval()
    deconv = Deconvolution(model)
    input_tensor = torch.tensor(example, dtype=torch.float32).unsqueeze(0)
    if device is not None:
        input_tensor = input_tensor.to(device)
    attributions = deconv.attribute(input_tensor, target=target_class)
    return attributions.squeeze(0).detach().cpu().numpy()


def explain_prediction_with_guided_backprop(
    model: torch.nn.Module,
    example: np.ndarray,
    target_class: int,
    device: torch.device | None = None,
) -> np.ndarray:
    model.eval()
    guided_bp = GuidedBackprop(model)
    input_tensor = torch.tensor(example, dtype=torch.float32).unsqueeze(0)
    if device is not None:
        input_tensor = input_tensor.to(device)
    attributions = guided_bp.attribute(input_tensor, target=target_class)
    return attributions.squeeze(0).detach().cpu().numpy()


def explain_prediction_with_input_x_gradient(
    model: torch.nn.Module,
    example: np.ndarray,
    target_class: int,
    device: torch.device | None = None,
) -> np.ndarray:
    model.eval()
    input_x_grad = InputXGradient(model)
    input_tensor = torch.tensor(example, dtype=torch.float32).unsqueeze(0)
    if device is not None:
        input_tensor = input_tensor.to(device)
    attributions = input_x_grad.attribute(input_tensor, target=target_class)
    return attributions.squeeze(0).detach().cpu().numpy()


def explain_prediction_with_integrated_gradients(
    model: torch.nn.Module,
    example: np.ndarray,
    target_class: int,
    device: torch.device | None = None,
) -> np.ndarray:
    model.eval()
    integrated_gradients = IntegratedGradients(model)
    input_tensor = torch.tensor(example, dtype=torch.float32).unsqueeze(0)
    if device is not None:
        input_tensor = input_tensor.to(device)
    attributions = integrated_gradients.attribute(
        input_tensor, target=target_class, n_steps=200
    )
    return attributions.squeeze(0).detach().cpu().numpy()


def display_explainibility(
    example_values: np.ndarray,
    attributions_values: np.ndarray,
    example_minimal_value: float = 100.0,
    atributions_minimal_value: float = 0.0001,
    example_alpha_visibility: float = 0.1,
    figsize: tuple[int, int] = (12, 12),
    z_slize_size: int = 10,
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

    plt.tight_layout()
    plt.show()
