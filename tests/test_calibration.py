import torch

from src.evaluation.calibration import expected_calibration_error, fit_temperature


def test_temperature_and_ece():
    logits = torch.randn(2,5,8,8)
    labels = torch.randint(0,5,(2,8,8))
    scaler = fit_temperature(logits, labels, max_pixels=100)
    assert scaler.temperature.item() > 0
    probs = scaler.transform(logits).softmax(1).permute(0,2,3,1).reshape(-1,5)
    ece = expected_calibration_error(probs, labels.reshape(-1))
    assert 0.0 <= ece <= 1.0


def test_temperature_stays_positive_and_handles_ignore_index():
    logits = torch.randn(1, 5, 8, 8)
    labels = torch.full((1, 8, 8), 255, dtype=torch.long)
    labels[:, :4, :] = torch.randint(0, 5, (1, 4, 8))
    scaler = fit_temperature(logits, labels, max_pixels=100)
    assert scaler.temperature.item() > 0
    assert torch.isfinite(scaler.transform(logits)).all()
