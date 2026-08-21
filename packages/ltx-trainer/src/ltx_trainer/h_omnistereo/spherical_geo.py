"""Spherical disparity for top-bottom ERP rigs (Eq. 1)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def spherical_disparity(
    theta_top: Tensor,
    theta_bottom: Tensor,
    rb: Tensor,
    baseline: Tensor | float,
) -> Tensor:
    """
    d = θ_b − θ_t = arctan(sin(θ_b) / (r_b/B + cos(θ_b)))  (paper Eq. 1).

    All tensors broadcast; angles in radians.
    """
    if isinstance(baseline, (int, float)):
        b = baseline
    else:
        b = baseline
    denom = rb / b + torch.cos(theta_bottom)
    return torch.atan(torch.sin(theta_bottom) / denom.clamp(min=1e-6))


def polar_angle_from_erp_row(v: Tensor, height: int) -> Tensor:
    """Map row index v ∈ [0,H) to polar angle θ ∈ [0, π] (equirectangular)."""
    return (v.float() + 0.5) / height * math.pi


def erp_row_grid(height: int, width: int, device: torch.device | None = None) -> Tensor:
    v = torch.arange(height, device=device, dtype=torch.float32)
    return polar_angle_from_erp_row(v, height).view(-1, 1).expand(height, width)
