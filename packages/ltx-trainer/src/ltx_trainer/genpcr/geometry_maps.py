"""Depth / equirectangular range maps for ControlNet conditioning (Sec. 3.3–3.4)."""

from __future__ import annotations

import torch
from torch import Tensor


def lidar_to_range_map(
    points: Tensor,
    *,
    height: int,
    width: int,
) -> Tensor:
    """
    Equirectangular range map Dequi ∈ R^{H×W×1} (Sec. 3.4.2, Eq. 6).

    points: N×3 in LiDAR coordinates.
    """
    if points.dim() != 2 or points.shape[1] != 3:
        raise ValueError("points must be N×3")
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = torch.linalg.norm(points, dim=1).clamp(min=1e-6)
    theta = torch.acos((z / r).clamp(-1.0, 1.0))
    phi = torch.atan2(y, x)
    ui = ((phi + torch.pi) / (2 * torch.pi) * width).long().clamp(0, width - 1)
    vi = (theta / torch.pi * height).long().clamp(0, height - 1)
    depth_map = torch.full((height, width), float("inf"), device=points.device)
    for i in range(points.shape[0]):
        u, v = int(ui[i]), int(vi[i])
        depth_map[v, u] = min(depth_map[v, u].item(), float(r[i]))
    depth_map = torch.where(torch.isfinite(depth_map), depth_map, torch.zeros_like(depth_map))
    return depth_map.unsqueeze(0)  # 1×H×W


def points_to_depth_map(
    points: Tensor,
    *,
    height: int,
    width: int,
    focal: float = 1.0,
) -> Tensor:
    """Perspective depth map stub from 3D points (depth-camera path)."""
    z = points[:, 2].clamp(min=1e-3)
    # scatter to grid by x,y projection (simplified pinhole)
    u = ((points[:, 0] / z * focal + 0.5) * width).long().clamp(0, width - 1)
    v = ((points[:, 1] / z * focal + 0.5) * height).long().clamp(0, height - 1)
    dmap = torch.zeros(height, width, device=points.device)
    for i in range(points.shape[0]):
        dmap[v[i], u[i]] = max(dmap[v[i], u[i]].item(), float(z[i]))
    return dmap.unsqueeze(0)
