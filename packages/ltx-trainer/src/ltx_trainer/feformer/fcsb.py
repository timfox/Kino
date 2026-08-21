"""Frequency-enabled Cross-scale Stem Bridge — FCSB (Sec. 3.5.1, Eq. 35–48)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.feformer.fft_utils import fft3, ifft3


class FCSB(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        half = max(channels // 2, 1)
        self.sp = nn.Conv3d(half, half, 1)
        self.restore = nn.Conv3d(channels, channels, 1)
        self.q = nn.Conv3d(channels, half, 1)
        self.k = nn.Conv3d(channels, half, 1)
        self.v = nn.Conv3d(half, half, 1)
        self.out = nn.Conv3d(half, channels, 1)

    def forward(self, x1: Tensor, x2: Tensor) -> tuple[Tensor, Tensor]:
        c = x2.shape[1]
        half = max(c // 2, 1)
        x1_2, x2_2 = x2[:, :half], x2[:, half : half + half]
        if x2_2.shape[1] < half:
            x2_2 = F.pad(x2_2, (0, 0, 0, 0, 0, 0, 0, half - x2_2.shape[1]))
        xf = fft3(x2_2).real
        xf = F.relu(self.sp(xf))
        x2_hat = self.restore(torch.cat([x1_2, xf], dim=1))

        x1_up = F.interpolate(x1_2, size=x1.shape[-3:], mode="trilinear", align_corners=False)
        q, k, v = self.q(x1), self.k(x1), self.v(x1_2)
        attn = ifft3(fft3(q) * fft3(k))
        attn = F.softmax(attn.flatten(2), dim=-1).view_as(attn)
        x1_hat = self.out(x1_up + attn * v)
        if x1_hat.shape[-3:] != x1.shape[-3:]:
            x1_hat = F.interpolate(x1_hat, size=x1.shape[-3:], mode="trilinear", align_corners=False)
        return x1_hat, x2_hat
