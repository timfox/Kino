"""Frozen MoGe-style encoder stub for pixel-aligned image tokens."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MoGeEncoderStub(nn.Module):
    """Lightweight stand-in for frozen MoGe ViT-L feature grid."""

    def __init__(self, in_ch: int = 4, feat_dim: int = 64) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, feat_dim, 3, padding=1),
            nn.GELU(),
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, rgba: Tensor) -> Tensor:
        return self.backbone(rgba)
