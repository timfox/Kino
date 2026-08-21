"""Frozen DINOv2-style encoder stub (Sec. 4.1)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FrozenDinoStub(nn.Module):
    """Lightweight frozen backbone approximating DINOv2-Base features."""

    def __init__(self, *, out_channels: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.Conv2d(128, out_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32 if out_channels >= 32 else 1, out_channels),
            nn.GELU(),
        )
        for p in self.stem.parameters():
            p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.stem(x)

    def train(self, mode: bool = True) -> FrozenDinoStub:
        super().train(mode)
        self.stem.eval()
        return self
