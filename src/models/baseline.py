from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SixChannelUNet(nn.Module):
    """Simple six-channel U-Net baseline: RGB pre + RGB post at input."""

    def __init__(self, num_damage_classes: int = 5):
        super().__init__()
        self.e1 = _Block(6, 32); self.e2 = _Block(32, 64); self.e3 = _Block(64, 128); self.b = _Block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.d3 = _Block(256 + 128, 128); self.d2 = _Block(128 + 64, 64); self.d1 = _Block(64 + 32, 32)
        self.building_head = nn.Conv2d(32, 2, 1)
        self.damage_head = nn.Conv2d(32, num_damage_classes, 1)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat([pre, post], dim=1)
        s1 = self.e1(x); s2 = self.e2(self.pool(s1)); s3 = self.e3(self.pool(s2)); x = self.b(self.pool(s3))
        x = F.interpolate(x, size=s3.shape[-2:], mode="bilinear", align_corners=False); x = self.d3(torch.cat([x, s3], dim=1))
        x = F.interpolate(x, size=s2.shape[-2:], mode="bilinear", align_corners=False); x = self.d2(torch.cat([x, s2], dim=1))
        x = F.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False); x = self.d1(torch.cat([x, s1], dim=1))
        x = F.interpolate(x, size=pre.shape[-2:], mode="bilinear", align_corners=False)
        return {"building_logits": self.building_head(x), "damage_logits": self.damage_head(x)}
