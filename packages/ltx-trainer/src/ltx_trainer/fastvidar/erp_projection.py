"""Fisheye ray → ERP lattice (Sec. III-A, Eq. 1–2)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def ray_to_erp_pixels(
    direction: Tensor,
    height: int,
    width: int,
) -> tuple[Tensor, Tensor]:
    """
    Unit ray d = (dx, dy, dz) with +z forward → ERP (x, y).

    λ = atan2(dx, dz), φ = arcsin(dy)
    x = (λ/2π + 1/2) W,  y = (1/2 − φ/π) H
    """
    dx, dy, dz = direction[..., 0], direction[..., 1], direction[..., 2]
    lam = torch.atan2(dx, dz)
    phi = torch.asin(dy.clamp(-1.0, 1.0))
    x = (lam / (2.0 * math.pi) + 0.5) * width
    y = (0.5 - phi / math.pi) * height
    return x, y


def erp_to_unit_ray(x: Tensor, y: Tensor, height: int, width: int) -> Tensor:
    """Inverse: ERP pixel → unit ray on S²."""
    lam = 2.0 * math.pi * (x.float() / width - 0.5)
    phi = math.pi * (0.5 - y.float() / height)
    return torch.stack(
        [
            torch.sin(lam) * torch.cos(phi),
            torch.sin(phi),
            torch.cos(lam) * torch.cos(phi),
        ],
        dim=-1,
    )


def erp_latitude_weight(height: int, device: torch.device | None = None) -> Tensor:
    """Row weights w(v) = cos φ(v) for ERP-area loss (Sec. III-D)."""
    v = torch.arange(height, device=device, dtype=torch.float32)
    phi = math.pi * (v + 0.5) / height - 0.5
    return torch.cos(phi)
