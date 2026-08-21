"""SpaFreq dual-stream classifier utilities (Sec. 3.2, Eq. 1–4)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def minmax_norm(x: Tensor) -> Tensor:
    lo = x.amin(dim=(-2, -1), keepdim=True)
    hi = x.amax(dim=(-2, -1), keepdim=True)
    return (x - lo) / (hi - lo + 1e-8)


def frequency_view(gray: Tensor, kernel_size: int = 9) -> Tensor:
    """Approximate Eq. (1): stack normalized low/high frequency bands (db4 proxy)."""
    if gray.dim() == 2:
        gray = gray.unsqueeze(0).unsqueeze(0)
    elif gray.dim() == 3:
        gray = gray.unsqueeze(1)
    pad = kernel_size // 2
    g = torch.tensor([0.25, 0.5, 0.25], device=gray.device, dtype=gray.dtype)
    g_h = g.view(1, 1, 1, -1)
    g_v = g.view(1, 1, -1, 1)
    c_a = F.conv2d(gray, g_h, padding=(0, pad))
    c_a = F.conv2d(c_a, g_v, padding=(pad, 0))
    gx = gray[..., :, 1:] - gray[..., :, :-1]
    gy = gray[..., 1:, :] - gray[..., :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    c_h = minmax_norm(gx.abs())
    c_v = minmax_norm(gy.abs())
    c_a = minmax_norm(c_a)
    if c_a.shape[-2:] != gray.shape[-2:]:
        c_a = F.interpolate(c_a, size=gray.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat([c_a, c_h, c_v], dim=1)


def fusion_weight(w_param: Tensor) -> Tensor:
    """Eq. (3): alpha = sigmoid(w_fusion)."""
    return torch.sigmoid(w_param)


def fuse_spatial_frequency(fs: Tensor, fw: Tensor, w_param: Tensor) -> Tensor:
    """Eq. (4): z = concat(alpha * fs, (1-alpha) * fw)."""
    alpha = fusion_weight(w_param)
    while alpha.dim() < fs.dim():
        alpha = alpha.unsqueeze(-1)
    return torch.cat([alpha * fs, (1.0 - alpha) * fw], dim=-1)


class SpaFreqHead(nn.Module):
    """Lightweight MLP head (1024->512) on fused embeddings."""

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class SpaFreqFusion(nn.Module):
    """Dual-stream fusion block; backbone features supplied externally."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.w_fusion = nn.Parameter(torch.tensor(0.0))

    def forward(self, f_spatial: Tensor, f_freq: Tensor) -> Tensor:
        return fuse_spatial_frequency(f_spatial, f_freq, self.w_fusion)
