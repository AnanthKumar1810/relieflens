import numpy as np

from src.dashboard.inference import predict_pair
from src.models.relieflens import ReliefLensNet


def test_predict_pair_preserves_original_image_shape():
    model = ReliefLensNet(pretrained=False)
    pre = np.zeros((65, 97, 3), dtype=np.uint8)
    post = np.zeros((65, 97, 3), dtype=np.uint8)
    result = predict_pair(model, pre, post, device="cpu", normalize="none", temperature=1.0)
    assert result["building_mask"].shape == (65, 97)
    assert result["damage_mask"].shape == (65, 97)
    assert result["damage_probs"].shape == (5, 65, 97)
    assert 0.0 <= result["priority"] <= 1.0


def test_predict_pair_rejects_mismatched_dimensions():
    model = ReliefLensNet(pretrained=False)
    pre = np.zeros((64, 64, 3), dtype=np.uint8)
    post = np.zeros((65, 64, 3), dtype=np.uint8)
    try:
        predict_pair(model, pre, post, device="cpu", normalize="none")
    except ValueError as exc:
        assert "identical dimensions" in str(exc)
    else:
        raise AssertionError("Mismatched pre/post dimensions must be rejected by default")
