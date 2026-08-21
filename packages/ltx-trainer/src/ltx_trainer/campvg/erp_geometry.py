"""Panoramic spherical coordinates (Sec. 3.2, Eq. 3–4)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def pixel_to_spherical_campvg(u: Tensor, v: Tensor, height: int, width: int) -> tuple[Tensor, Tensor]:
    """Eq. (3): φ = u/W·2π, θ = v/H·π."""
    phi = u.float() / width * (2.0 * math.pi)
    theta = v.float() / height * math.pi
    return phi, theta


def spherical_to_direction(phi: Tensor, theta: Tensor) -> Tensor:
    """Eq. (4): unit direction in camera frame."""
    return torch.stack(
        [
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta),
            torch.cos(theta) * torch.cos(phi),
        ],
        dim=-1,
    )


def direction_grid(height: int, width: int, device: torch.device | None = None) -> Tensor:
    """[H, W, 3] unit rays for ERP lattice."""
    v = torch.arange(height, device=device, dtype=torch.float32)
    u = torch.arange(width, device=device, dtype=torch.float32)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    phi, theta = pixel_to_spherical_campvg(uu, vv, height, width)
    return spherical_to_direction(phi, theta)
