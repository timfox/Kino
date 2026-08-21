"""ERP spherical geometry (Eq. 1–2, Wang et al. arXiv:2605.13169)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def pixel_to_yaw_pitch(u: Tensor, v: Tensor, height: int, width: int) -> tuple[Tensor, Tensor]:
    """
    Map ERP pixel indices to yaw λ and pitch φ (radians).

    λ = 2π(u/W − 1/2),  φ = π(1/2 − v/H).
    """
    lam = 2.0 * math.pi * (u.float() / width - 0.5)
    phi = math.pi * (0.5 - v.float() / height)
    return lam, phi


def unit_ray(lam: Tensor, phi: Tensor) -> Tensor:
    """Unit viewing direction r(λ, φ) = [cos φ sin λ, sin φ, cos φ cos λ]."""
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    sin_lam = torch.sin(lam)
    cos_lam = torch.cos(lam)
    x = cos_phi * sin_lam
    y = sin_phi
    z = cos_phi * cos_lam
    return torch.stack([x, y, z], dim=-1)


def patch_center_grid(
    height: int,
    width: int,
    patch_size: int,
    *,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Patch center (u, v) for a non-overlapping patch grid."""
    gh = height // patch_size
    gw = width // patch_size
    pv = torch.arange(gh, device=device, dtype=torch.float32) * patch_size + patch_size / 2
    pu = torch.arange(gw, device=device, dtype=torch.float32) * patch_size + patch_size / 2
    vv, uu = torch.meshgrid(pv, pu, indexing="ij")
    return uu.reshape(-1), vv.reshape(-1)


def patch_spherical_directions(
    height: int,
    width: int,
    patch_size: int,
    *,
    device: torch.device | None = None,
) -> Tensor:
    """[N, 3] unit rays at patch centers."""
    u, v = patch_center_grid(height, width, patch_size, device=device)
    lam, phi = pixel_to_yaw_pitch(u, v, height, width)
    return unit_ray(lam, phi)
