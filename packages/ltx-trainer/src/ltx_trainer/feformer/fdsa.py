"""Frequency-enhanced Dynamic Self-Attention — FDSA (Sec. 3.3, Eq. 3–14)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.feformer.fft_utils import fft3, frequency_band_decompose, ifft3


class FDSA(nn.Module):
    def __init__(self, channels: int, *, dw_kernel: int = 7) -> None:
        super().__init__()
        pad = dw_kernel // 2
        self.dw = nn.Conv3d(channels, channels, dw_kernel, padding=pad, groups=channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)
        self.freq_fc = nn.Sequential(
            nn.Linear(channels * 3, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        feat = self.dw(x)
        q, k, v = self.qkv(feat).chunk(3, dim=1)
        qf, kf = fft3(q), fft3(k)
        attn_freq = qf * kf
        attn = ifft3(attn_freq)
        attn = F.softmax(attn.flatten(2), dim=-1).view_as(attn)
        x_sa = attn * v

        x_freq = fft3(x_sa)
        bands = frequency_band_decompose(x_freq, bands=3)
        z = torch.stack([b.mean(dim=(-3, -2, -1)) for b in bands], dim=1)  # B,3,C
        z = z.flatten(1)
        w = self.freq_fc(z).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return self.proj(w * x_sa)
