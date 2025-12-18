from captum.attr import LayerGradCam, GuidedGradCam, LayerAttribution
import torch
import numpy as np


def explain_prediction_with_grad_cam(
    model: torch.nn.Module,
    last_layer: torch.nn.Module,
    example: np.ndarray,
    target_class: int,
    device: torch.device | None = None,
):
    model.eval()
    layer_gc = LayerGradCam(model, last_layer)
    input_tensor = torch.tensor(example, dtype=torch.float32).unsqueeze(0)
    if device is not None:
        input_tensor = input_tensor.to(device)
    return LayerAttribution.interpolate(
        layer_gc.attribute(input_tensor, target=target_class),
        example.shape[-3:],
        interpolate_mode="trilinear",
    )


def explain_prediction_with_guided_grad_cam(
    model: torch.nn.Module,
    last_layer: torch.nn.Module,
    example: np.ndarray,
    target_class: int,
    device: torch.device | None = None,
):
    model.eval()
    layer_gc = GuidedGradCam(model, last_layer)
    input_tensor = torch.tensor(example, dtype=torch.float32).unsqueeze(0)
    if device is not None:
        input_tensor = input_tensor.to(device)
    return layer_gc.attribute(input_tensor, target=target_class)
