"""Heading-aligned surface normals (Eq. 2, Fig. 2)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def pixel_longitude_alpha(width: int, u: int | Tensor) -> Tensor | float:
    """Longitude α ∈ [0, 2π) for column u in ERP image of given width."""
    if isinstance(u, Tensor):
        return (u.float() + 0.5) / width * 2.0 * math.pi
    return (float(u) + 0.5) / width * 2.0 * math.pi


def longitude_grid(width: int, height: int, device: torch.device | None = None) -> Tensor:
    """Per-pixel longitude α [H, W]."""
    u = torch.arange(width, device=device, dtype=torch.float32)
    alpha = (u + 0.5) / width * 2.0 * math.pi
    return alpha.unsqueeze(0).expand(height, -1)


def camera_to_heading_aligned(n_cam: Tensor, alpha: Tensor) -> Tensor:
    """Apply Eq. (2): rotate camera-frame normal by −α about Z (heading alignment)."""
    if n_cam.dim() == 3:
        n_cam = n_cam.unsqueeze(0)
    b, _, h, w = n_cam.shape
    if alpha.dim() == 0:
        a = alpha.view(1, 1, 1, 1).expand(b, 1, h, w)
    elif alpha.dim() == 2:
        a = alpha.view(1, 1, h, w).expand(b, 1, h, w)
    elif alpha.dim() == 3:
        a = alpha.unsqueeze(1)
    else:
        a = alpha if alpha.shape[1] == 1 else alpha.unsqueeze(1)
    ca, sa = torch.cos(a), torch.sin(a)
    nx, ny, nz = n_cam[:, 0:1], n_cam[:, 1:2], n_cam[:, 2:3]
    nha_x = ca * nx - sa * ny
    nha_y = sa * nx + ca * ny
    n_ha = torch.cat([nha_x, nha_y, nz], dim=1)
    return torch.nn.functional.normalize(n_ha, dim=1, eps=1e-6)


def heading_aligned_to_camera(n_ha: Tensor, alpha: Tensor) -> Tensor:
    """Inverse of heading alignment (rotate by +α)."""
    if n_ha.dim() == 3:
        n_ha = n_ha.unsqueeze(0)
    b, _, h, w = n_ha.shape
    if alpha.dim() == 2:
        a = alpha.view(1, 1, h, w).expand(b, 1, h, w)
    elif alpha.dim() == 3:
        a = alpha.unsqueeze(1)
    else:
        a = alpha if alpha.shape[1] == 1 else alpha.unsqueeze(1)
    ca, sa = torch.cos(a), torch.sin(a)
    nx, ny, nz = n_ha[:, 0:1], n_ha[:, 1:2], n_ha[:, 2:3]
    nc_x = ca * nx + sa * ny
    nc_y = -sa * nx + ca * ny
    return torch.nn.functional.normalize(torch.cat([nc_x, nc_y, nz], dim=1), dim=1, eps=1e-6)


def angular_loss_deg(pred: Tensor, target: Tensor) -> Tensor:
    """Squared angular loss L_normal = ∠(n, n̄)² in radians² (paper Eq. 3)."""
    p = torch.nn.functional.normalize(pred, dim=1, eps=1e-6)
    t = torch.nn.functional.normalize(target, dim=1, eps=1e-6)
    cos = (p * t).sum(dim=1).clamp(-1.0, 1.0)
    ang = torch.acos(cos)
    return (ang**2).mean()
