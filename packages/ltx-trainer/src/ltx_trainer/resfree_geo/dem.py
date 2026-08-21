"""Diffusion-based density-equalizing maps (Sec. 3.3)."""

from __future__ import annotations

import torch
from torch import Tensor


def fick_velocity(rho: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    """v = −∇ρ / ρ (Eq. 7); returns vx, vy from central differences."""
    rho_s = rho.clamp(min=eps)
    grad_y, grad_x = torch.gradient(rho_s)
    return -grad_x / rho_s, -grad_y / rho_s


def smooth_l1(x: Tensor, y: Tensor) -> Tensor:
    """Huber / smooth L1 used in DEM loss (Sec. 5.2)."""
    diff = (x - y).abs()
    quad = 0.5 * diff**2
    lin = diff - 0.5
    return torch.where(diff < 1.0, quad, lin).mean()


def dem_area_loss(pop_per_cell: Tensor, target_density: Tensor, area: Tensor) -> Tensor:
    """L(pi/Ai - p/A) style density matching on triangles (stub)."""
    ratio = pop_per_cell / area.clamp(min=1e-8)
    target = target_density / area.sum() * area
    return smooth_l1(ratio, target / area.clamp(min=1e-8))


def normalize_density(p: Tensor) -> Tensor:
    """Unit mass: ∫ p = 1."""
    p_pos = p - p.min() + 1e-3
    return p_pos / p_pos.sum()


def sin_cos_test_density(h: int = 51, w: int = 51) -> Tensor:
    """Eq. 28."""
    xs = torch.linspace(0, 1, w)
    ys = torch.linspace(0, 1, h)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    p = 2.0 + torch.sin(2 * torch.pi * xx) * torch.cos(2 * torch.pi * yy)
    return normalize_density(p)


def ring_test_density(h: int = 51, w: int = 51) -> Tensor:
    """Eq. 31."""
    xs = torch.linspace(0, 1, w)
    ys = torch.linspace(0, 1, h)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    r2 = (xx - 0.5) ** 2 + (yy - 0.5) ** 2
    p = torch.exp(-((torch.sqrt(r2) - 1.0) ** 2) / (2 * 0.5**2))
    return normalize_density(p)


def jacobian_det_2d(u: Tensor, v: Tensor) -> Tensor:
    """det(Jf) for u(x,y), v(x,y) via torch.gradient."""
    du_dy, du_dx = torch.gradient(u)
    dv_dy, dv_dx = torch.gradient(v)
    return du_dx * dv_dy - du_dy * dv_dx
