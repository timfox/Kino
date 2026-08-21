"""Panoramic projection formats (Sec. 2.3, supplementary)."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


def erp_pixel_to_spherical(u: Tensor, v: Tensor, width: int, height: int) -> tuple[Tensor, Tensor]:
    """ERP: φ ∈ [-π, π], θ ∈ [0, π] per survey supplementary."""
    phi = u.float() / width * (2.0 * math.pi) - math.pi
    theta = v.float() / height * math.pi
    return phi, theta


def spherical_to_unit(phi: Tensor, theta: Tensor) -> Tensor:
    return torch.stack(
        [
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ],
        dim=-1,
    )


def cubemap_face_uv(x: Tensor, y: Tensor, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Dominant-axis cubemap face index and face UV in [-1, 1]."""
    ax = torch.stack([x.abs(), y.abs(), z.abs()], dim=-1)
    face = ax.argmax(dim=-1)
    # front (+z) stub
    u = x / z.clamp(min=1e-6)
    v = y / z.clamp(min=1e-6)
    return face, u, v


def panini_scale(phi: Tensor, d: float = 1.0) -> Tensor:
    """Panini horizontal compression factor S = (d + 1) / (d + cos φ)."""
    return (d + 1.0) / (d + torch.cos(phi).clamp(min=1e-6))


def projection_catalogue() -> dict[str, Any]:
    return {
        "spherical": "Distortion-free direction sphere; foundation for all planar maps.",
        "erp": "Longitude–latitude unwrap; default ODI format in this survey.",
        "cubemap": "Six 90° faces; reduced polar stretch vs ERP.",
        "tangent": "Gnomonic patches on polyhedron; reuses perspective backbones.",
        "polyhedron": "Icosahedron/octahedron faces; near-uniform sampling.",
        "panini": "Wide-FoV compromise preserving vertical/radial lines.",
    }
