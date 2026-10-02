from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SharedResNet18(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = resnet18(weights=weights)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]


class ReliefLensNet(nn.Module):
    """Siamese change-detection U-Net with building and damage heads."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.1, num_damage_classes: int = 5):
        super().__init__()
        self.encoder = SharedResNet18(pretrained=pretrained)
        fused = [192, 384, 768, 1536]
        self.dec4 = ConvBlock(fused[3], 256, dropout)
        self.dec3 = ConvBlock(256 + fused[2], 128, dropout)
        self.dec2 = ConvBlock(128 + fused[1], 64, dropout)
        self.dec1 = ConvBlock(64 + fused[0], 32, dropout)
        self.refine = ConvBlock(32, 32, dropout)
        self.building_head = nn.Conv2d(32, 2, 1)
        self.damage_head = nn.Conv2d(32, num_damage_classes, 1)

    @staticmethod
    def _fuse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b, torch.abs(a - b)], dim=1)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> dict[str, torch.Tensor]:
        pre_f = self.encoder(pre)
        post_f = self.encoder(post)
        f1, f2, f3, f4 = [self._fuse(a, b) for a, b in zip(pre_f, post_f)]
        x = self.dec4(f4)
        x = F.interpolate(x, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat([x, f3], dim=1))
        x = F.interpolate(x, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, f2], dim=1))
        x = F.interpolate(x, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, f1], dim=1))
        x = F.interpolate(x, size=pre.shape[-2:], mode="bilinear", align_corners=False)
        x = self.refine(x)
        return {"building_logits": self.building_head(x), "damage_logits": self.damage_head(x)}
