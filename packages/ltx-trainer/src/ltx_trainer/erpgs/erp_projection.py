"""ERP camera model and distortion-aware weights (Eq. 2–4, 14)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def erp_lon_lat(mu_c: Tensor) -> tuple[Tensor, Tensor]:
    """
    Project camera-frame point to lon/lat (Eq. 2).
    mu_c: [..., 3] with x right, y down, z forward.
    """
    x, y, z = mu_c[..., 0], mu_c[..., 1], mu_c[..., 2]
    lon = torch.atan2(x, z)
    norm = mu_c.norm(dim=-1).clamp_min(1e-6)
    lat = torch.asin((y / norm).clamp(-1.0, 1.0))
    return lon, lat


def erp_pixel_coords(lon: Tensor, lat: Tensor, h: int, w: int) -> Tensor:
    """μ^p from lon/lat (Eq. 3). Returns [..., 2] as (u, v)."""
    u = (w / (2 * math.pi)) * (lon + math.pi)
    v = (h / (2 * math.pi)) * (2 * lat + math.pi)
    return torch.stack([u, v], dim=-1)


def distortion_weight_map(h: int, w: int, device: torch.device | None = None) -> Tensor:
    """
    Per-pixel cos(latitude) weight W (Eq. 14 integral discretized).
    Higher latitude → smaller solid angle per pixel.
    """
    row = torch.arange(h, device=device, dtype=torch.float32)
    lat = (row + 0.5) * (math.pi / h) - math.pi / 2
    cos_lat = torch.cos(lat).clamp_min(1e-3)
    return cos_lat.view(h, 1).expand(h, w)


def jacobian_erp_stub(mu_c: Tensor) -> Tensor:
    """Simplified 2×2 image covariance factor from ERP Jacobian (Eq. 4 stub)."""
    lon, lat = erp_lon_lat(mu_c)
    scale = torch.cos(lat).clamp_min(0.1).unsqueeze(-1)
    return torch.diag_embed(scale.expand(*scale.shape[:-1], 2))
