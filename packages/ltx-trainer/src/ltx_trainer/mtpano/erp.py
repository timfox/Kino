"""ERP geometry and token mixer (Eq. 2, Sec. 3.2.1)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def spherical_ray_direction(phi: Tensor, theta: Tensor) -> Tensor:
    """Unit ray d = [cos φ cos θ, sin φ, cos φ sin θ]^T (supp. Eq. 8)."""
    x = torch.cos(phi) * torch.cos(theta)
    y = torch.sin(phi)
    z = torch.cos(phi) * torch.sin(theta)
    return torch.stack([x, y, z], dim=-1)


def latitude_weight(phi: Tensor) -> Tensor:
    """w(φ) = |sin(φ)| for ERP mixer (Eq. 2)."""
    return torch.abs(torch.sin(phi))


class ERPTokenMixer(nn.Module):
    """Latitude-adaptive dual-kernel mixing (Eq. 2)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_wide = nn.Conv2d(channels, channels, kernel_size=(3, 9), padding=(1, 4))

    def forward(self, x: Tensor, phi_map: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        if phi_map.shape[-2:] != (h, w):
            phi_map = F.interpolate(
                phi_map.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        lat_w = latitude_weight(phi_map).unsqueeze(0).unsqueeze(0)
        local = self.conv3(x)
        wide = self.conv_wide(x)
        return (1 - lat_w) * local + lat_w * wide


def erp_phi_grid(height: int, width: int, device: torch.device | None = None) -> Tensor:
    """Per-pixel latitude φ for ERP of size H×W."""
    phi, _ = erp_spherical_coords(height, width, device=device)
    return phi


def erp_spherical_coords(
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """ERP latitude φ and longitude θ per pixel (supp. Eq. 8 uses u∈[-1,1], v∈[-1,1])."""
    v = torch.linspace(-0.5, 0.5, height, device=device, dtype=dtype) * math.pi
    u = torch.linspace(-0.5, 0.5, width, device=device, dtype=dtype) * 2 * math.pi
    phi = v.view(-1, 1).expand(height, width)
    theta = u.view(1, -1).expand(height, width)
    return phi, theta


def film_condition_map(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Ray direction (3) + (θ, φ) positions → (5, H, W) for FiLM."""
    phi, theta = erp_spherical_coords(height, width, device=device, dtype=dtype)
    ray = spherical_ray_direction(phi, theta).permute(2, 0, 1)
    return torch.cat([ray, theta.unsqueeze(0), phi.unsqueeze(0)], dim=0)
