"""Spherical ↔ Cartesian coordinate transforms (Eq. 1, Fig. 2)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def cartesian_to_spherical(s: Tensor) -> Tensor:
    """S (B×3×H×W or B×N×3) → u=(θ, φ) same layout, last dim 2."""
    sx, sy, sz = s[..., 0], s[..., 1], s[..., 2]
    norm = torch.linalg.norm(s, dim=-1).clamp_min(1e-8)
    sx, sy, sz = sx / norm, sy / norm, sz / norm
    theta = torch.atan2(sx, sz)
    phi = torch.asin(sy.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    return torch.stack([theta, phi], dim=-1)


def spherical_to_cartesian(u: Tensor) -> Tensor:
    """u=(θ, φ) → unit sphere S."""
    theta, phi = u[..., 0], u[..., 1]
    sx = torch.sin(theta) * torch.cos(phi)
    sy = torch.sin(phi)
    sz = torch.cos(theta) * torch.cos(phi)
    return torch.stack([sx, sy, sz], dim=-1)


def erp_grid_to_spherical(h: int, w: int, device: torch.device | None = None) -> Tensor:
    """2D ERP pixel grid → spherical coords (H×W×2)."""
    ys = torch.linspace(-math.pi / 2, math.pi / 2, h, device=device)
    xs = torch.linspace(-math.pi, math.pi, w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1)


def erp_grid_to_cartesian(h: int, w: int, device: torch.device | None = None) -> Tensor:
    return spherical_to_cartesian(erp_grid_to_spherical(h, w, device))


def pi_inv_erp_grid(feat_h: int, feat_w: int, device: torch.device | None = None) -> Tensor:
    """π⁻¹ on feature grid for spherical positional embedding (Eq. 9)."""
    return erp_grid_to_cartesian(feat_h, feat_w, device)
