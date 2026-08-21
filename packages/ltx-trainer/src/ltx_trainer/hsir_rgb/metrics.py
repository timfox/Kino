"""HS restoration metrics: PSNR, SAM, SSIM (arXiv:2605.24769 evaluation protocol)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def hs_psnr(x_hat: Tensor, x: Tensor, *, data_range: float = 1.0, eps: float = 1e-10) -> float:
    """Mean PSNR over all bands and spatial locations (MSE in [0, data_range])."""
    if x_hat.shape != x.shape:
        raise ValueError(f"Shape mismatch {x_hat.shape} vs {x.shape}")
    mse = F.mse_loss(x_hat, x).item()
    if mse < eps:
        return 99.0
    return float(10.0 * math.log10((data_range**2) / mse))


def spectral_angle_mapper(x_hat: Tensor, x: Tensor, *, eps: float = 1e-8) -> float:
    """Mean spectral angle (degrees) over spatial locations."""
    if x_hat.shape != x.shape:
        raise ValueError(f"Shape mismatch {x_hat.shape} vs {x.shape}")
    dot = (x_hat * x).sum(dim=1)
    n1 = x_hat.norm(dim=1).clamp(min=eps)
    n2 = x.norm(dim=1).clamp(min=eps)
    cos = (dot / (n1 * n2)).clamp(-1.0 + eps, 1.0 - eps)
    ang = torch.acos(cos)
    return float(ang.mean().item() * 180.0 / math.pi)


def hs_ssim_band_mean(x_hat: Tensor, x: Tensor, *, window_size: int = 11, data_range: float = 1.0) -> float:
    """Average 2D SSIM over spectral bands (Gaussian window, channel-wise)."""
    if x_hat.shape != x.shape or x_hat.dim() != 4:
        raise ValueError("Expected [B,C,H,W]")
    b, c, h, w = x_hat.shape
    if h < window_size or w < window_size:
        return float(1.0 - F.l1_loss(x_hat, x).item() / data_range)

    def ssim_2d(a: Tensor, b: Tensor) -> Tensor:
        # a,b [1,1,H,W]
        mu_a = a.mean(dim=(-2, -1), keepdim=True)
        mu_b = b.mean(dim=(-2, -1), keepdim=True)
        sigma_a = a.var(dim=(-2, -1), unbiased=False, keepdim=True)
        sigma_b = b.var(dim=(-2, -1), unbiased=False, keepdim=True)
        sigma_ab = ((a - mu_a) * (b - mu_b)).mean(dim=(-2, -1), keepdim=True)
        c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
        num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
        den = (mu_a**2 + mu_b**2 + c1) * (sigma_a + sigma_b + c2)
        return (num / den).mean()

    win = min(window_size, h, w)
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return float(1.0 - F.mse_loss(x_hat, x).item())

    total = 0.0
    for bi in range(b):
        for ci in range(c):
            xa = x_hat[bi : bi + 1, ci : ci + 1]
            xb = x[bi : bi + 1, ci : ci + 1]
            total += float(ssim_2d(xa, xb).item())
    return total / max(b * c, 1)
