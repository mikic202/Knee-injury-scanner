import torch
import numpy as np

from src.model_architecture.resnet3d.resnet import get_resnet3d, FocalLoss
from src.model_architecture.cnn_clasifier.cnn_clasifier import CnnKneeClassifier
from src.model_architecture.SAE.SAE import SaeEncoder, SaeDecoder


def test_sae_encoder_decoder_forward_and_sparsity():
    batch = torch.randn(4, 64)
    encoder = SaeEncoder(input_size=64, hidden_size=32, max_hidden_features=8)
    decoder = SaeDecoder(hidden_size=32, output_size=64)

    h = encoder(batch)
    assert h.shape == (4, 32)
    nz = (h != 0).sum(dim=1)
    assert all(nz <= 8)

    out = decoder(h)
    assert out.shape == (4, 64)


def test_resnet_device_and_forward():
    device = torch.device('cpu')
    model = get_resnet3d(num_classes=4, in_channels=1, device=device)
    assert next(model.parameters()).device == device

    x = torch.randn(2, 1, 32, 64, 64)
    model.eval()
    out = model(x)
    assert out.shape == (2, 4)


def test_focal_loss_variants():
    logits = torch.randn(6, 4)
    targets = torch.randint(0, 4, (6,))

    fl = FocalLoss(alpha=None, gamma=0.0, reduction='mean')
    m = fl(logits, targets)
    assert m.dim() == 0

    fl2 = FocalLoss(alpha=torch.tensor([1.0, 2.0, 0.5, 1.0]), gamma=2.0, reduction='none')
    out = fl2(logits, targets)
    assert out.shape == (6,)


def test_cnn_classifier_forward_and_shapes():
    model = CnnKneeClassifier(input_channels=1, num_classes=3)
    inp = torch.randn(1, 1, 64, 64, 64)
    out = model(inp)
    assert out.shape == (1, 3)

