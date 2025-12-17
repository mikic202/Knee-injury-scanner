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
