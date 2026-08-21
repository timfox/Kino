"""DPT-style segmentation decoder (Sec. 4.1)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.weatherproof.classes import NUM_CLASSES


class DPTSegHead(nn.Module):
    """Task-specific head trained while backbone stays frozen."""

    def __init__(self, in_channels: int = 256, *, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32 if in_channels >= 32 else 1, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.GroupNorm(16 if in_channels >= 32 else 1, in_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(in_channels // 2, num_classes, 1)

    def forward(self, feat: Tensor, *, out_size: tuple[int, int] | None = None) -> Tensor:
        x = self.refine(feat)
        if out_size is not None:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return self.classifier(x)
