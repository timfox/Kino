"""ERP ↔ spherical geometry (Eq. 3–4)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def pixel_to_spherical(x: Tensor, y: Tensor, height: int, width: int) -> tuple[Tensor, Tensor]:
    """
    Eq. (3): θ = 2πx/W − π,  φ = πy/H − π/2.
    """
    theta = 2.0 * math.pi * x.float() / width - math.pi
    phi = math.pi * y.float() / height - math.pi / 2.0
    return theta, phi


def spherical_distance_haversine(
    theta1: Tensor,
    phi1: Tensor,
    theta2: Tensor,
    phi2: Tensor,
    *,
    radius: float = 1.0,
) -> Tensor:
    """
    Eq. (4): great-circle distance on the sphere.
    """
    dphi = phi2 - phi1
    dtheta = theta2 - theta1
    a = torch.sin(dphi / 2) ** 2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(dtheta / 2) ** 2
    return 2.0 * radius * torch.asin(torch.sqrt(a.clamp(0.0, 1.0)))


def patch_spherical_coords(
    height: int,
    width: int,
    patch_size: int,
    *,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Patch-center (θ, φ) for an ERP grid."""
    gh = height // patch_size
    gw = width // patch_size
    py = torch.arange(gh, device=device, dtype=torch.float32) * patch_size + patch_size / 2
    px = torch.arange(gw, device=device, dtype=torch.float32) * patch_size + patch_size / 2
    yy, xx = torch.meshgrid(py, px, indexing="ij")
    return pixel_to_spherical(xx.reshape(-1), yy.reshape(-1), height, width)
