import torch

from src.models.relieflens import ReliefLensNet


def test_model_forward_shapes():
    model = ReliefLensNet(pretrained=False)
    x = torch.randn(2, 3, 64, 64)
    out = model(x, x)
    assert out["building_logits"].shape == (2, 2, 64, 64)
    assert out["damage_logits"].shape == (2, 5, 64, 64)


def test_baseline_forward_shapes():
    from src.models.baseline import SixChannelUNet
    model = SixChannelUNet()
    x = torch.randn(2, 3, 64, 64)
    out = model(x, x)
    assert out["building_logits"].shape == (2, 2, 64, 64)
    assert out["damage_logits"].shape == (2, 5, 64, 64)
