"""Heteroscedastic AR(1) camera noise (Sec. 3.1, Eq. 1–2)."""

from __future__ import annotations

import random

import torch
from torch import Tensor

from ltx_trainer.diffhdr.config import NOISE_AR1_RHO, NOISE_SIGMA_C_RANGE, NOISE_SIGMA_S_RANGE


def heteroscedastic_noise(
    linear: Tensor,
    *,
    sigma_s: float | None = None,
    sigma_c: float | None = None,
    rho: float = NOISE_AR1_RHO,
) -> Tensor:
    """
    Apply temporally correlated camera noise to video [T,C,H,W] or [B,T,C,H,W].
    """
    if linear.dim() == 4:
        return _noise_video(linear, sigma_s, sigma_c, rho)
    if linear.dim() == 5:
        outs = [_noise_video(linear[:, i], sigma_s, sigma_c, rho) for i in range(linear.shape[1])]
        return torch.stack(outs, dim=1)
    raise ValueError(f"expected 4D or 5D tensor, got {linear.dim()}D")


def _noise_video(video: Tensor, sigma_s: float | None, sigma_c: float | None, rho: float) -> Tensor:
    t, c, h, w = video.shape
    rng = random.Random(0)
    ss = sigma_s if sigma_s is not None else rng.uniform(*NOISE_SIGMA_S_RANGE)
    sc = sigma_c if sigma_c is not None else rng.uniform(*NOISE_SIGMA_C_RANGE)
    out = video.clone()
    eps_prev = torch.zeros(c, h, w, device=video.device, dtype=video.dtype)
    for i in range(t):
        u = torch.randn(c, h, w, device=video.device, dtype=video.dtype)
        eps = rho * eps_prev + (1.0 - rho**2) ** 0.5 * u
        l = out[i].clamp(min=0.0)
        noise = torch.sqrt(l * ss + sc) * eps
        out[i] = (l + noise).clamp(min=0.0)
        eps_prev = eps
    return out
