"""Omnidirectional neighbor pixel selection (Eq. 9–10)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def tangent_offsets(h: int, w: int) -> Tensor:
    """Offsets (tx, ty) on tangent plane at origin (Sec. 3.2)."""
    tx = math.tan(2 * math.pi / w)
    ty = math.tan(2 * math.pi / h)
    return torch.tensor(
        [
            [tx, 0.0],
            [-tx, 0.0],
            [0.0, ty],
            [0.0, -ty],
        ]
    )


def erp_neighbor_angles(theta: Tensor, phi: Tensor, tx: float, ty: float) -> tuple[Tensor, Tensor]:
    """
    Map tangent-plane offset to ERP angles (Eq. 9–10).
    theta, phi: center pixel spherical angles (radians).
    """
    rho = math.sqrt(tx * tx + ty * ty)
    nu = math.atan(rho)
    sin_nu = math.sin(nu)
    cos_nu = math.cos(nu)
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    num_phi = cos_nu * sin_phi + ty * sin_nu * cos_phi
    phi_n = torch.asin((num_phi / rho).clamp(-1, 1)) if rho > 1e-8 else phi

    denom = rho * cos_phi * cos_nu - ty * sin_phi * sin_nu
    theta_n = theta + torch.atan((tx * sin_nu) / denom.clamp_min(1e-6))
    return theta_n, phi_n
