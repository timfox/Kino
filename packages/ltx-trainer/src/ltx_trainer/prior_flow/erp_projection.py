"""ERP spherical mapping (Sec. 3.1, Eq. 1)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def erp_pixel_to_spherical(u: Tensor, v: Tensor, width: int, height: int) -> tuple[Tensor, Tensor]:
    """Map ERP pixel (u,v) to longitude θ and latitude ϕ (Eq. 1)."""
    theta = (u + 0.5) * (2.0 * math.pi / width) - math.pi
    phi = math.pi / 2.0 - (v + 0.5) * (math.pi / height)
    return theta, phi


def spherical_area_weights(height: int, width: int, device: torch.device | None = None) -> Tensor:
    """Per-pixel ω_j ∝ cos(latitude) for spherical L1 (Eq. 17)."""
    v = torch.arange(height, device=device, dtype=torch.float32)
    phi = math.pi / 2.0 - (v + 0.5) * (math.pi / height)
    w_row = torch.cos(phi).clamp_min(1e-6)
    return w_row.view(1, 1, height, 1).expand(1, 1, height, width)
