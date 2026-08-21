"""Distortion-aware weighted averaging (Eq. 9–10)."""

from __future__ import annotations

import torch
from torch import Tensor


def perspective_weights(h: int, w: int, tau: float, device: torch.device | None = None) -> Tensor:
    """W^I_jk = exp(-||u_jk|| / τ)."""
    ys = torch.linspace(-1, 1, h, device=device)
    xs = torch.linspace(-1, 1, w, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    dist = torch.sqrt(gx * gx + gy * gy)
    return torch.exp(-dist / tau)


def fuse_views(
    view_latents: list[Tensor],
    view_weights: list[Tensor],
) -> Tensor:
    """Hadamard-weighted sum then normalize (Eq. 7 stub)."""
    num = torch.zeros_like(view_latents[0])
    den = torch.zeros_like(view_latents[0][:, :1])
    for lat, w in zip(view_latents, view_weights, strict=True):
        w4 = w.unsqueeze(0).unsqueeze(0)
        num = num + lat * w4
        den = den + w4
    return num / den.clamp_min(1e-6)
