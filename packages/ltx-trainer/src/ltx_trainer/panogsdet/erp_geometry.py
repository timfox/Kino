"""Equirectangular depth → 3D points (PanoGSDet Eq. 1)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def depth_map_to_points(depth: Tensor) -> Tensor:
    """
    Map depth D [B,H,W] or [B,1,H,W] to camera-centric 3D points [B,H,W,3].

    Paper (1-based u,v): θ=(u/H−0.5)π, η=(1−v/W)2π, then x,y,z from d.
    """
    if depth.dim() == 4:
        depth = depth.squeeze(1)
    b, h, w = depth.shape
    u = torch.arange(h, device=depth.device, dtype=depth.dtype).view(1, h, 1)
    v = torch.arange(w, device=depth.device, dtype=depth.dtype).view(1, 1, w)
    # 1-based indices as in the paper
    u1 = u + 1.0
    v1 = v + 1.0
    theta = (u1 / h - 0.5) * math.pi
    eta = (1.0 - v1 / w) * 2.0 * math.pi
    sin_e, cos_e = torch.sin(eta), torch.cos(eta)
    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    x = depth * sin_e
    y = depth * cos_t * cos_e
    z = depth * cos_e * sin_t
    return torch.stack([x, y, z], dim=-1)


def flatten_points(points: Tensor) -> Tensor:
    """[B,H,W,3] → [B,N,3] with N=H*W."""
    b, h, w, _ = points.shape
    return points.reshape(b, h * w, 3)
