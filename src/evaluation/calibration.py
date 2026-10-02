from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import LBFGS


class TemperatureScaler(nn.Module):
    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        if float(init_temperature) <= 0:
            raise ValueError("init_temperature must be > 0")
        self.log_temperature = nn.Parameter(torch.log(torch.tensor([float(init_temperature)])))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp_min(1e-3)

    @torch.no_grad()
    def transform(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.detach().clamp_min(1e-3)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_pixels: int = 100_000,
                    init_temperature: float = 1.0, ignore_index: int = 255) -> TemperatureScaler:
    logits = logits.detach().float().reshape(-1, logits.shape[1])
    labels = labels.detach().reshape(-1).long()
    valid = labels != ignore_index
    logits, labels = logits[valid], labels[valid]
    if logits.numel() == 0:
        return TemperatureScaler(init_temperature)
    if logits.shape[0] > max_pixels:
        idx = torch.linspace(0, logits.shape[0] - 1, max_pixels).long()
        logits, labels = logits[idx], labels[idx]
    scaler = TemperatureScaler(init_temperature).to(logits.device)
    optimizer = LBFGS([scaler.log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")
    nll = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = nll(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, bins: int = 15,
                               ignore_index: int = 255) -> float:
    probs = probs.detach().float()
    labels = labels.detach().long()
    conf, pred = probs.max(dim=1)
    valid = labels != ignore_index
    conf, pred, labels = conf[valid], pred[valid], labels[valid]
    if labels.numel() == 0:
        return float("nan")
    ece = torch.zeros((), device=probs.device)
    edges = torch.linspace(0, 1, bins + 1, device=probs.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            acc = (pred[in_bin] == labels[in_bin]).float().mean()
            avg_conf = conf[in_bin].mean()
            ece += in_bin.float().mean() * (avg_conf - acc).abs()
    return float(ece.item())
