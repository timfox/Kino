"""Plücker ray maps for extrinsic camera conditioning (Sec. 3.2, Fig. 4).

Maps encode oriented lines as ``(ω, m)`` with ``m = o × ω`` for ray origin ``o`` and
unit direction ``ω``. For same-view intrinsic-only edits, use zero maps (paper: zero pose rays).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def plucker_from_rays(directions: Tensor, origins: Tensor) -> Tensor:
    """``directions``, ``origins``: ``[..., 3]`` → Plücker ``[..., 6]`` = concat(ω̂, o×ω̂)."""
    if directions.shape != origins.shape:
        raise ValueError(f"Shape mismatch {directions.shape} vs {origins.shape}")
    omega = F.normalize(directions, dim=-1, eps=1e-8)
    moment = torch.linalg.cross(origins, omega, dim=-1)
    return torch.cat([omega, moment], dim=-1)


def zeros_plucker_map(height: int, width: int, *, device: torch.device | None = None) -> Tensor:
    """Zero Plücker map for locked extrinsics (same-view style / intrinsic edit)."""
    return torch.zeros(height, width, 6, device=device)


def plucker_map_pinhole(
    intrinsics: Tensor,
    c2w: Tensor,
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
) -> Tensor:
    """World-space Plücker map from pinhole ``K`` and camera-to-world ``4×4``.

    ``intrinsics``: ``[3, 3]``, ``c2w``: ``[4, 4]`` (OpenCV-style: x right, y down, z forward).
    """
    if intrinsics.shape != (3, 3) or c2w.shape != (4, 4):
        raise ValueError("intrinsics must be [3,3], c2w must be [4,4]")
    dev = device or intrinsics.device
    yy, xx = torch.meshgrid(
        torch.arange(height, device=dev, dtype=torch.float32),
        torch.arange(width, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(xx)
    pix = torch.stack([xx, yy, ones], dim=-1)  # [H,W,3]
    k_inv = torch.linalg.inv(intrinsics.to(dev))
    dirs_cam = torch.einsum("ij,...j->...i", k_inv, pix)  # [H,W,3]
    r = c2w[:3, :3].to(dev)
    t = c2w[:3, 3].to(dev)
    dirs_w = torch.einsum("ij,...j->...i", r, dirs_cam)
    origins = t.expand_as(dirs_w)
    return plucker_from_rays(dirs_w, origins)
