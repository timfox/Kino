"""Distortion-Aware Spectral Modulator (Sec. III-C, Fig. 2b–d)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HFEM(nn.Module):
    """High-Frequency Enhancement Module (Fig. 2c)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gate = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        h = F.relu(self.conv1(x))
        return self.gate(self.conv3(h)) * x


class LFSM(nn.Module):
    """Low-Frequency Stabilization Module (Fig. 2d)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.shape
        w = self.mlp(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * w


def laplacian_highpass(x: Tensor) -> Tensor:
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    c = x.shape[1]
    k = kernel.expand(c, 1, 3, 3)
    return F.conv2d(x, k, padding=1, groups=c)


def gaussian_lowpass(x: Tensor, sigma: float = 1.0) -> Tensor:
    return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)


class DASM(nn.Module):
    """Dual-frequency spectral modulation with gated fusion (Eq. 3)."""

    def __init__(self, channels: int, text_dim: int) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.hfem = HFEM(channels)
        self.lfsm = LFSM(channels)
        self.lambda_h = nn.Parameter(torch.tensor(0.5))
        self.lambda_l = nn.Parameter(torch.tensor(0.5))
        self.ch_gate = nn.Linear(text_dim, channels)
        self.spatial_gate = nn.Conv2d(channels, 1, 1)
        self.reaggregate = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
        )

    def forward(self, fv: Tensor, ft: Tensor) -> Tensor:
        """
        fv: [B, L, C] visual tokens; ft: [B, C_cls, C] text (mean-pooled for gate).
        Returns [B, L, C].
        """
        b, l, c = fv.shape
        text_ctx = ft.mean(dim=1, keepdim=True).expand(-1, l, -1)
        f_prime, _ = self.cross_attn(fv, text_ctx, text_ctx)
        h = int(l**0.5) if int(l**0.5) ** 2 == l else max(8, int(l**0.5))
        w = l // h
        if h * w != l:
            pad = h * w - l
            f_map = F.pad(f_prime.transpose(1, 2), (0, pad)).view(b, c, h, w)
        else:
            f_map = f_prime.transpose(1, 2).view(b, c, h, w)
        fh = self.hfem(laplacian_highpass(f_map))
        fl = self.lfsm(gaussian_lowpass(f_map))
        g_ch = torch.sigmoid(self.ch_gate(ft.mean(dim=1))).view(b, c, 1, 1)
        g_sp = torch.sigmoid(self.spatial_gate(f_map))
        fused = f_map
        for branch, lam in ((fh, self.lambda_h), (fl, self.lambda_l)):
            fused = fused + lam * g_ch * g_sp * branch
        flat = fused.flatten(2).transpose(1, 2)
        if flat.shape[1] > l:
            flat = flat[:, :l]
        elif flat.shape[1] < l:
            flat = F.pad(flat, (0, 0, 0, l - flat.shape[1]))
        return self.reaggregate(flat) + f_prime
