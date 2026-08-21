"""BEV homography calibration from uncalibrated surveillance video (Sec. III-B)."""

from __future__ import annotations

import torch
from torch import Tensor


def _normalize_points(pts: Tensor) -> tuple[Tensor, Tensor]:
    """Hartley normalization for DLT stability."""
    mean = pts.mean(dim=0)
    shifted = pts - mean
    scale = (2.0**0.5) / shifted.norm(dim=1).mean().clamp(min=1e-6)
    t = torch.eye(3, device=pts.device, dtype=pts.dtype)
    t[0, 0] = scale
    t[1, 1] = scale
    t[0, 2] = -scale * mean[0]
    t[1, 2] = -scale * mean[1]
    hom = torch.cat([pts, torch.ones(pts.shape[0], 1, device=pts.device, dtype=pts.dtype)], dim=1)
    normalized = (t @ hom.T).T[:, :2]
    return normalized, t


def estimate_homography(image_pts: Tensor, world_pts: Tensor) -> Tensor:
    """Estimate 3×3 homography mapping image → BEV world (planar)."""
    if image_pts.shape != world_pts.shape or image_pts.shape[1] != 2:
        raise ValueError("image_pts and world_pts must be (N, 2)")
    n = image_pts.shape[0]
    if n < 4:
        raise ValueError("need at least 4 point correspondences")

    src_n, t_src = _normalize_points(image_pts)
    dst_n, t_dst = _normalize_points(world_pts)

    a_rows: list[Tensor] = []
    for i in range(n):
        x, y = src_n[i]
        u, v = dst_n[i]
        a_rows.append(torch.tensor([-x, -y, -1, 0, 0, 0, u * x, u * y, u], device=image_pts.device))
        a_rows.append(torch.tensor([0, 0, 0, -x, -y, -1, v * x, v * y, v], device=image_pts.device))
    a = torch.stack(a_rows)
    _, _, vh = torch.linalg.svd(a)
    h = vh[-1].reshape(3, 3)
    h = torch.linalg.inv(t_dst) @ h @ t_src
    return h / h[2, 2]


def apply_homography(pts: Tensor, h: Tensor) -> Tensor:
    """Map (N, 2) image points to BEV."""
    ones = torch.ones(pts.shape[0], 1, device=pts.device, dtype=pts.dtype)
    ph = torch.cat([pts, ones], dim=1) @ h.T
    return ph[:, :2] / ph[:, 2:3]


def reprojection_error(image_pts: Tensor, world_pts: Tensor, h: Tensor) -> float:
    """Mean pixel error after world→image round-trip via H⁻¹."""
    h_inv = torch.linalg.inv(h)
    reproj = apply_homography(world_pts, h_inv)
    return float(torch.norm(reproj - image_pts, dim=1).mean())


def reconstruction_error_m(world_pred: Tensor, world_gt: Tensor) -> float:
    """Mean metric-plane error in meters."""
    return float(torch.norm(world_pred - world_gt, dim=1).mean())
