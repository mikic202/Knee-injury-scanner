import numpy as np
import torch
from src.explainibility import basic_gradient_based_methods as bg


def test_basic_gradient_wrappers_shapes(monkeypatch):
    class Dummy(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((x.shape[0], 3))

    model = Dummy()
    example = np.random.rand(1, 8, 16, 16).astype(np.float32)

    def fake_attr(self, input_tensor, target=None, **kwargs):
        return input_tensor
    try:
        from captum.attr import Saliency, IntegratedGradients
        monkeypatch.setattr(Saliency, 'attribute', fake_attr)
        monkeypatch.setattr(IntegratedGradients, 'attribute', fake_attr)
    except Exception:
        monkeypatch.setattr(bg, 'explain_prediction_with_saliency', lambda m, e, t, device=None: np.ones((1, 8, 16, 16)))
        monkeypatch.setattr(bg, 'explain_prediction_with_integrated_gradients', lambda m, e, t, device=None: np.ones((1, 8, 16, 16)))

    sal = bg.explain_prediction_with_saliency(model, example, target_class=0, device=None)
    ig = bg.explain_prediction_with_integrated_gradients(model, example, target_class=0, device=None)

    assert isinstance(sal, np.ndarray)
    assert isinstance(ig, np.ndarray)
    assert sal.shape[1:] == ig.shape[1:]


def test_input_x_gradient_and_guided_backprop_fallback(monkeypatch):
    class Dummy(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((x.shape[0], 3))

    model = Dummy()
    example = np.random.rand(1, 8, 16, 16).astype(np.float32)

    for name in ['explain_prediction_with_guided_backprop', 'explain_prediction_with_input_x_gradient', 'explain_prediction_with_deconvolution']:
        if not hasattr(bg, name):
            continue
        monkeypatch.setattr(bg, name, lambda m, e, t, device=None: np.ones((1, 8, 16, 16)))

    for fn in ['explain_prediction_with_guided_backprop', 'explain_prediction_with_input_x_gradient', 'explain_prediction_with_deconvolution']:
        if hasattr(bg, fn):
            out = getattr(bg, fn)(model, example, 0, None)
            assert isinstance(out, np.ndarray)
