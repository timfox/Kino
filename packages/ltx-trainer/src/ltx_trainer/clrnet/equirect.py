"""Equirectangular depth image projection (Sec. III-B)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def cartesian_to_spherical(x: Tensor, y: Tensor, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    r = torch.sqrt(x * x + y * y + z * z + 1e-8)
    theta = torch.atan2(y, x)
    phi = torch.asin((z / r).clamp(-1.0, 1.0))
    return theta, phi, r


def equirect_project_points(
    points: Tensor,
    *,
    height: int,
    width: int,
    extra_channels: Tensor | None = None,
) -> Tensor:
    """
    Project [B, N, 3] (+ optional [B, N, C_extra]) to [B, C, H, W] depth image.

    Channel 0 is range r; extra channels are max-pooled per pixel.
    """
    b, n, _ = points.shape
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    theta, phi, r = cartesian_to_spherical(x, y, z)
    theta_norm = (theta + math.pi) / (2 * math.pi)
    phi_norm = (phi + math.pi / 2) / math.pi
    u = (theta_norm * width).long().clamp(0, width - 1)
    v = ((1.0 - phi_norm) * height).long().clamp(0, height - 1)

    c_extra = 0 if extra_channels is None else extra_channels.shape[-1]
    out = torch.zeros(b, 1 + c_extra, height, width, device=points.device, dtype=points.dtype)
    for bi in range(b):
        for idx in range(n):
            ui, vi = u[bi, idx].item(), v[bi, idx].item()
            out[bi, 0, vi, ui] = r[bi, idx]
            if extra_channels is not None:
                out[bi, 1:, vi, ui] = extra_channels[bi, idx]
    return out


def stack_lidar_depth(points: Tensor, intensity: Tensor, *, height: int, width: int) -> Tensor:
    extra = intensity.unsqueeze(-1) if intensity.dim() == 2 else intensity
    return equirect_project_points(points, height=height, width=width, extra_channels=extra)


def stack_radar_depth(
    points: Tensor,
    rcs: Tensor,
    velocity: Tensor,
    time_ch: Tensor,
    *,
    height: int,
    width: int,
) -> Tensor:
    extra = torch.stack([rcs, velocity, time_ch], dim=-1)
    return equirect_project_points(points, height=height, width=width, extra_channels=extra)
