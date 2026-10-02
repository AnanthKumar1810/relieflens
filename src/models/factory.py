from __future__ import annotations

from src.models.baseline import SixChannelUNet
from src.models.relieflens import ReliefLensNet


def build_model(model_type: str, cfg: dict, *, pretrained_override: bool | None = None):
    model_type = model_type.lower()
    if model_type == "baseline":
        return SixChannelUNet(cfg["model"]["num_damage_classes"])
    if model_type == "siamese":
        pretrained = cfg["model"]["pretrained"] if pretrained_override is None else pretrained_override
        return ReliefLensNet(pretrained=pretrained, dropout=cfg["model"]["dropout"], num_damage_classes=cfg["model"]["num_damage_classes"])
    raise ValueError(f"Unknown model_type={model_type!r}. Use 'siamese' or 'baseline'.")
