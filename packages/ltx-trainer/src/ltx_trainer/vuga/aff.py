"""Adaptive Feature Fusion + Spatial Distortion-aware Attention (Sec. III-D)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SDAModule(nn.Module):
    """Spatial Distortion-aware Attention (Eq. 7–8)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, 1)
        self.gate_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.gate_off = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        f = self.proj(x)
        u = torch.sigmoid(self.gate_off(self.gate_dw(f)))
        sgu = u * f
        return self.out(sgu) + x


class AFFModule(nn.Module):
    """Top-down progressive fusion of F1–F4_cmp (Eq. 9)."""

    def __init__(self, cmp_dim: int = 512) -> None:
        super().__init__()
        self.sda34 = SDAModule(cmp_dim)
        self.sda234 = SDAModule(cmp_dim)
        self.sda_all = SDAModule(cmp_dim)

    def forward(self, cmp_feats: list[Tensor]) -> Tensor:
        f1, f2, f3, f4 = cmp_feats
        f43 = self.sda34(f3 + F.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False))
        f234 = self.sda234(f2 + F.interpolate(f43, size=f2.shape[-2:], mode="bilinear", align_corners=False))
        return self.sda_all(f1 + F.interpolate(f234, size=f1.shape[-2:], mode="bilinear", align_corners=False))
