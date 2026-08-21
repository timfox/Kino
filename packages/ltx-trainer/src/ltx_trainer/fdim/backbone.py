"""Multi-scale backbone (ResNet-18 style, lightweight for tests)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _Stage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.gn(self.conv(x)), inplace=True)


class MultiScaleBackbone(nn.Module):
    """Extract 4-scale pyramids {conv2x..conv5x} from RGB frames."""

    def __init__(self, in_ch: int = 3, widths: tuple[int, int, int, int] = (32, 64, 128, 256)) -> None:
        super().__init__()
        w1, w2, w3, w4 = widths
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, w1, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(min(8, w1), w1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage2 = _Stage(w1, w1)
        self.stage3 = _Stage(w1, w2, stride=2)
        self.stage4 = _Stage(w2, w3, stride=2)
        self.stage5 = _Stage(w3, w4, stride=2)

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        f2 = self.stage2(x)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)
        return [f2, f3, f4, f5]
