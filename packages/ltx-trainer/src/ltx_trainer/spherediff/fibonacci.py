"""Fibonacci lattice on S² (Sec. 3.1)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ltx_trainer.spherediff.config import SphereDiffConfig


def fibonacci_sphere(n: int, device: torch.device | None = None) -> Tensor:
    """
    Nearly uniform points on unit sphere [N, 3].
    Golden-angle spiral (Hardin–Saff style).
    """
    i = torch.arange(n, device=device, dtype=torch.float32)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - (2.0 * i + 1.0) / n
    r = torch.sqrt(1.0 - y * y)
    theta = phi * i
    x = torch.cos(theta) * r
    z = torch.sin(theta) * r
    return torch.stack([x, y, z], dim=-1)


def spherical_latents_init(cfg: SphereDiffConfig, device: torch.device | None = None) -> tuple[Tensor, Tensor]:
    """S = {(d_i, f_i)} — directions [N,3] and features [N,C]."""
    dirs = fibonacci_sphere(cfg.num_latents, device=device)
    feats = torch.randn(cfg.num_latents, cfg.latent_dim, device=device) * 0.02
    return dirs, feats
