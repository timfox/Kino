"""Frequency-decomposed Gating MLP — FGMLP (Sec. 3.4, Eq. 15–24)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FGMLP(nn.Module):
    def __init__(self, channels: int, *, mlp_ratio: int = 4, dw_kernel: int = 7) -> None:
        super().__init__()
        hidden = channels * mlp_ratio
        self.fc1 = nn.Conv3d(channels, hidden, 1)
        self.fc2 = nn.Conv3d(hidden, channels, 1)
        self.dropout = nn.Dropout(0.1)
        pad = dw_kernel // 2
        self.dw = nn.Conv3d(channels * 2, 2, 1)
        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv3d(channels // 4, channels, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = F.gelu(self.fc1(x))
        gate = h * F.relu6(h)
        out = self.dropout(self.fc2(self.dropout(gate)))

        k = torch.sigmoid(self.kernel_gen(out))
        x_low = out * k
        x_high = out - x_low

        avp = out.mean(dim=(-3, -2, -1), keepdim=True)
        mp = out.amax(dim=(-3, -2, -1), keepdim=True)
        w = torch.sigmoid(self.dw(torch.cat([avp, mp], dim=1)))
        w1, w2 = w[:, :1], w[:, 1:2]
        return w1 * x_low + w2 * x_high
