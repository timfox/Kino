"""Exposure-invariant φ (align) and merge ψ features."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _luma(rgb: Tensor) -> Tensor:
    w = rgb.new_tensor([0.2126, 0.7152, 0.0722]).view(3, 1, 1)
    return (rgb * w).sum(dim=0, keepdim=True)


def align_features(img: Tensor) -> Tensor:
    """φ features: luma + log-luma + Sobel edges (4×H×W)."""
    lum = _luma(img.clamp(1e-4, 1.0))
    log_lum = torch.log(lum)
    gx = lum[:, :, 1:] - lum[:, :, :-1]
    gy = lum[:, 1:, :] - lum[:, :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    return torch.cat([lum, log_lum, gx.abs(), gy.abs()], dim=0)


def merge_features(img: Tensor) -> Tensor:
    """ψ features: RGB mean + luma + local variance proxy (4×H×W)."""
    lum = _luma(img.clamp(0.0, 1.0))
    mean_rgb = img.mean(dim=0, keepdim=True)
    var = F.avg_pool2d((img - mean_rgb).pow(2).sum(dim=0, keepdim=True), 3, stride=1, padding=1)
    return torch.cat([mean_rgb.expand_as(lum), lum, var, img.max(dim=0, keepdim=True).values], dim=0)
