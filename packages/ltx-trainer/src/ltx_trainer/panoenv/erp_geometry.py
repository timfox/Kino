"""ERP spherical mapping and 3D centroid projection (Eq. 1–2)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def erp_pixel_to_spherical(px: Tensor, py: Tensor, width: int, height: int) -> tuple[Tensor, Tensor]:
    """λ, ϕ from ERP pixel (Eq. 1)."""
    lam = (px / width - 0.5) * 2.0 * math.pi
    phi = -(py / height - 0.5) * math.pi
    return lam, phi


def spherical_to_cartesian(depth: Tensor, lam: Tensor, phi: Tensor) -> Tensor:
    """3D point P = [x, y, z]^T (Eq. 2)."""
    x = -depth * torch.cos(phi) * torch.sin(lam)
    y = depth * torch.sin(phi)
    z = -depth * torch.cos(phi) * torch.cos(lam)
    return torch.stack([x, y, z], dim=-1)


def dominant_cubemap_face(lam: Tensor, phi: Tensor) -> Tensor:
    """Map unit direction to one of six cubemap faces (front/left/right/back/top/bottom)."""
    dirs = torch.stack(
        [
            -torch.cos(phi) * torch.sin(lam),  # +X right
            torch.sin(phi),  # +Y up
            -torch.cos(phi) * torch.cos(lam),  # -Z forward
        ],
        dim=-1,
    )
    abs_d = dirs.abs()
    face = abs_d.argmax(dim=-1)
    return face
