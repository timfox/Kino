"""Multi-scale ResNet50 encoder stub (Eq. 2)."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class MultiScaleEncoder(nn.Module):
    def __init__(self, ch: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, ch, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
        )
        self.coarse = nn.Sequential(nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.ReLU())
        self.fine = nn.Sequential(nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.ReLU())

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.stem(x)
        coarse = self.coarse(h)
        fine = self.fine(coarse)
        return coarse, fine
