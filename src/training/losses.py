from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.xbd import IGNORE_INDEX


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = logits.softmax(dim=1)[:, 1]
        tgt = target.float()
        valid = torch.ones_like(tgt, dtype=torch.bool)
        inter = (prob[valid] * tgt[valid]).sum()
        denom = prob[valid].sum() + tgt[valid].sum()
        return 1.0 - (2 * inter + self.smooth) / (denom + self.smooth)


class FocalCrossEntropy(nn.Module):
    def __init__(self, class_weights: list[float], gamma: float = 2.0, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if not class_weights or any(float(w) <= 0 for w in class_weights):
            raise ValueError("class_weights must contain positive values")
        self.register_buffer("weights", torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Focal loss: derive p_t from the *unweighted* cross-entropy, then
        # apply the configured alpha/class weight separately. Weighting the
        # CE before exp(-CE) changes p_t and no longer implements standard
        # focal cross-entropy.
        valid = target != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        flat_logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
        flat_target = target.reshape(-1)
        flat_valid = valid.reshape(-1)
        valid_logits = flat_logits[flat_valid]
        valid_target = flat_target[flat_valid]
        ce = F.cross_entropy(valid_logits, valid_target, reduction="none")
        pt = torch.exp(-ce)
        alpha = self.weights.to(logits.device)[valid_target]
        return (alpha * (1.0 - pt).pow(self.gamma) * ce).mean()


class ReliefLensLoss(nn.Module):
    def __init__(self, damage_class_weights: list[float], focal_gamma: float = 2.0,
                 building_weight: float = 1.0, damage_weight: float = 1.0):
        super().__init__()
        self.building_weight = building_weight
        self.damage_weight = damage_weight
        self.building_ce = nn.CrossEntropyLoss()
        self.building_dice = BinaryDiceLoss()
        self.damage = FocalCrossEntropy(damage_class_weights, focal_gamma)

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        b_ce = self.building_ce(outputs["building_logits"], batch["building"])
        b_dice = self.building_dice(outputs["building_logits"], batch["building"])
        d = self.damage(outputs["damage_logits"], batch["damage"])
        total = self.building_weight * (b_ce + b_dice) + self.damage_weight * d
        return {"total": total, "building_ce": b_ce.detach(), "building_dice": b_dice.detach(), "damage_focal": d.detach()}
