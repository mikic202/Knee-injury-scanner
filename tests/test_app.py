import numpy as np
import torch
import pickle
import io

from src.web_app import main as web_main
from src.model_architecture.resnet3d.resnet import get_resnet3d


def test_normalize_and_plot_helpers():
    a = np.zeros((8, 8), dtype=np.float32)
    img = web_main._normalize_for_display(a)
    assert img.dtype == np.uint8
    assert img.shape == a.shape

    vol = np.random.rand(16, 32, 32).astype(np.float32)
    model = get_resnet3d(num_classes=3, in_channels=1)
    model.eval()
    device = torch.device('cpu')
    pred_class, probs = web_main.predict_knee_diagnosis(model, device, vol, target_shape=(8, 16, 16))
    assert isinstance(pred_class, int)
    assert abs(sum(probs.values()) - 1.0) < 1e-4


def test_explain_buttons_monkeypatched(monkeypatch):
    vol = np.random.rand(16, 32, 32).astype(np.float32)

    def fake_ig(model, example, target_class, device=None):
        return np.ones((1, 16, 32, 32), dtype=np.float32)

    def fake_sal(model, example, target_class, device=None):
        return np.full((1, 16, 32, 32), 2.0, dtype=np.float32)

    monkeypatch.setattr(web_main, 'explain_prediction_with_integrated_gradients', fake_ig)
    monkeypatch.setattr(web_main, 'explain_prediction_with_saliency', fake_sal)

    # Use a dummy model that returns a valid classification output shape
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], 3)

    model = DummyModel()
    device = torch.device('cpu')

    ig = web_main.explain_integrated_gradients_stream(model, vol, device)
    sal = web_main.explain_saliency_stream(model, vol, device)
    assert ig.shape == (32, 32)
    assert sal.shape == (32, 32)


def test_load_model_fallback_on_bad_path(monkeypatch):
    def fake_load(path, map_location=None):
        raise RuntimeError('bad')

    monkeypatch.setattr(torch, 'load', fake_load)
    model, device = web_main.load_model('/non/existent/path.pt')
    assert model is None and device is None
    

def test_normalize_for_display_constant_and_range():
    a = np.zeros((8, 8), dtype=np.float32)
    img = web_main._normalize_for_display(a)
    assert img.dtype == np.uint8
    assert img.max() == 128

    b = np.linspace(0, 1, 16).reshape(4, 4).astype(np.float32)
    img2 = web_main._normalize_for_display(b)
    assert img2.dtype == np.uint8
    assert img2.max() == 255


def test_predict_knee_diagnosis_and_get_text():
    vol = np.random.rand(16, 32, 32).astype(np.float32)
    device = torch.device('cpu')
    model = get_resnet3d(num_classes=3, in_channels=1)
    model.eval()

    predicted_class, probs = web_main.predict_knee_diagnosis(model, device, vol, target_shape=(8, 16, 16))
    assert isinstance(predicted_class, int)
    assert all(k in probs for k in ["Zdrowe kolano (0)", "ACL częściowo zerwane (1)", "ACL całkowicie zerwane (2)"])
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-4

    assert "Kolano jest zdrowe" in web_main.get_diagnosis_text(0)
    assert "ACL częściowo" in web_main.get_diagnosis_text(1)
    assert "ACL całkowicie" in web_main.get_diagnosis_text(2)