"""Equirectangular projection (Eq. 1)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def sphere_to_erp(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    *,
    height: int,
    width: int,
) -> tuple[Tensor, Tensor]:
    """Map unit-sphere direction to ERP pixel coords (u, v)."""
    phi = torch.atan2(y, x)
    theta = torch.acos(z.clamp(-1.0, 1.0))
    u = (phi + math.pi) / (2 * math.pi) * width
    v = theta / math.pi * height
    return u, v


def erp_uv_grid(height: int, width: int, device: torch.device | None = None) -> tuple[Tensor, Tensor]:
    """Normalized longitude/latitude grids for PE."""
    v = torch.arange(height, device=device, dtype=torch.float32) + 0.5
    u = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    theta = v * (math.pi / height)
    phi = u * (2 * math.pi / width) - math.pi
    return phi, theta
