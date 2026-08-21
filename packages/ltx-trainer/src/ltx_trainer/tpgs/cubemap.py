"""Cubemap sampling ERP ↔ unit cube (Eq. 4–5, Sec. III-B)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

FACE_NAMES = ("front", "back", "left", "right", "up", "down")


def cartesian_to_spherical(x: Tensor, y: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
    """θ = arctan(x,z), φ = arctan(y, sqrt(x²+z²))."""
    theta = torch.atan2(x, z)
    phi = torch.atan2(y, torch.sqrt(x * x + z * z).clamp_min(1e-8))
    return theta, phi


def spherical_to_erp_uv(theta: Tensor, phi: Tensor, h: int, w: int) -> tuple[Tensor, Tensor]:
    """Eq. 5: longitude/latitude → ERP pixel coords."""
    u = (theta / (2 * math.pi) + 0.5) * w - 0.5
    v = (-phi / math.pi + 0.5) * h - 0.5
    return u, v


def face_grid(face: str, n: int, device: torch.device | None = None) -> Tensor:
    """Regular grid on cube face in [-0.5, 0.5]² → (N, 3) unit directions."""
    t = torch.linspace(-0.5, 0.5, n, device=device)
    a, b = torch.meshgrid(t, t, indexing="ij")
    if face == "front":
        x, y, z = a, b, torch.full_like(a, 0.5)
    elif face == "back":
        x, y, z = -a, b, torch.full_like(a, -0.5)
    elif face == "left":
        x, y, z = torch.full_like(a, -0.5), a, b
    elif face == "right":
        x, y, z = torch.full_like(a, 0.5), a, b
    elif face == "up":
        x, y, z = a, torch.full_like(a, 0.5), -b
    else:  # down
        x, y, z = a, torch.full_like(a, -0.5), b
    pts = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    return pts / pts.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def padded_grid_extent(h: int, p: int) -> tuple[float, float]:
    """Eq. 8: extended sampling range on face."""
    lo = -0.5 - 2 * p / h
    hi = 0.5 + 2 * p / h
    return lo, hi


def padded_fov_rad(h: int, p: int) -> float:
    """Eq. 9: fov' = 2πp/H + π/2."""
    return 2 * math.pi * p / h + math.pi / 2
