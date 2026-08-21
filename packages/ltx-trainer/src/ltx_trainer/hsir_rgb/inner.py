"""Placeholder inner denoiser D^l_sigma (DRUNet stand-in for smoke tests)."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


def awgn_inner_denoiser(sigma: float) -> Callable[[Tensor], Tensor]:
    """Lightweight Gaussian smoothing prior (not DRUNet weights)."""

    def _denoise(z: Tensor) -> Tensor:
        if z.dim() != 4:
            raise ValueError("z must be [B,c,H,W]")
        k = max(3, int(2 * round(sigma * 10) + 1))
        if k % 2 == 0:
            k += 1
        pad = k // 2
        return F.avg_pool2d(F.pad(z, [pad] * 4, mode="reflect"), k, stride=1)

    return _denoise


def rgb_band_teaser_denoise(z: Tensor, *, sigma: float = 0.1) -> Tensor:
    """Fig. 1 stand-in: smooth AWGN-like restoration on RGB latent groups."""
    return awgn_inner_denoiser(sigma)(z).clamp(0.0, 1.0)
