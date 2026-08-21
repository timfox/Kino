"""Fusion UNet + classical Mertens baseline (Talegaonkar et al. Sec. 4)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def bracket_to_linear(bracket_bncHW: Tensor, *, gamma: float = 2.2) -> Tensor:
    """γ-encoded bracket ``[B,N,C,H,W]`` → linear ``[B,N,C,H,W]``."""
    return bracket_bncHW.clamp(0.0, 1.0).pow(gamma)


def fuse_bracket(bracket_lin_bncHW: Tensor, weights_bnHW: Tensor) -> Tensor:
    """Weighted linear fusion ``[B,C,H,W]`` (eq. 8)."""
    if bracket_lin_bncHW.ndim != 5 or weights_bnHW.ndim != 4:
        raise ValueError("Expected bracket [B,N,C,H,W] and weights [B,N,H,W]")
    w = weights_bnHW.unsqueeze(2)
    return (bracket_lin_bncHW * w).sum(dim=1)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FusionUNet(nn.Module):
    """Lightweight UNet: N×RGB in → per-pixel softmax weights over N exposures."""

    def __init__(self, num_frames: int, *, base: int = 32) -> None:
        super().__init__()
        self.num_frames = num_frames
        in_ch = num_frames * 3
        self.enc1 = _ConvBlock(in_ch, base)
        self.enc2 = _ConvBlock(base, base * 2)
        self.pool = nn.MaxPool2d(2)
        self.mid = _ConvBlock(base * 2, base * 2)
        self.dec2 = _ConvBlock(base * 4, base)
        self.head = nn.Conv2d(base, num_frames, 1)

    def forward(self, bracket_lin_bncHW: Tensor) -> Tensor:
        b, n, c, h, w = bracket_lin_bncHW.shape
        if n != self.num_frames:
            raise ValueError(f"Expected N={self.num_frames}, got {n}")
        x = bracket_lin_bncHW.reshape(b, n * c, h, w)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        m = self.mid(self.pool(e2))
        d2 = F.interpolate(m, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = d1 + e1
        logits = self.head(d1)
        return F.softmax(logits, dim=1)


def mertens_fusion(bracket_lin_bncHW: Tensor, *, gamma_mertens: float = 1.0) -> Tensor:
    """Classical Mertens-style weights (contrast + saturation proxy) for ablation."""
    b, n, c, h, w = bracket_lin_bncHW.shape
    scores: list[Tensor] = []
    for i in range(n):
        frame = bracket_lin_bncHW[:, i]
        gray = frame.mean(dim=1, keepdim=True)
        gx = F.pad(gray, (0, 1, 0, 0))[:, :, :, 1:] - F.pad(gray, (1, 0, 0, 0))[:, :, :, :-1]
        gy = F.pad(gray, (0, 0, 0, 1))[:, :, 1:, :] - F.pad(gray, (0, 0, 1, 0))[:, :, :-1, :]
        contrast = (gx.abs() + gy.abs()).squeeze(1)
        sat = frame.std(dim=1)
        well = 1.0 - (2.0 * (gray.squeeze(1) - 0.5).abs())
        s = (contrast + 1e-6) * (sat + 1e-6) * (well.clamp(0.0, 1.0) + 1e-6)
        scores.append(s.pow(gamma_mertens))
    w = torch.stack(scores, dim=1)
    w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return fuse_bracket(bracket_lin_bncHW, w)
