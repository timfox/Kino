"""Per-group 8-DOF homographies (Sec. 3.4.1)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def identity_homography(device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.eye(3, device=device, dtype=dtype)


def homography_from_similarity(
    *,
    tx: float = 0.0,
    ty: float = 0.0,
    theta_rad: float = 0.0,
    scale: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Similarity transform embedded in 3×3 homography (subset of 8-DOF)."""
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    a = scale * c
    b = scale * s
    return torch.tensor(
        [[a, -b, tx], [b, a, ty], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )


def apply_homography_points(xy: Tensor, h_mat: Tensor) -> Tensor:
    """Map N×2 points with 3×3 homography."""
    if xy.ndim != 2 or xy.shape[-1] != 2:
        raise ValueError("xy must be (N, 2)")
    ones = torch.ones(xy.shape[0], 1, device=xy.device, dtype=xy.dtype)
    hom = torch.cat([xy, ones], dim=-1)
    out = hom @ h_mat.T
    return out[:, :2] / out[:, 2:3].clamp(min=1e-8)


def compose_path_motion(
    canonical_xy: Tensor,
    *,
    center: Tensor,
    local_delta: Tensor,
    homography: Tensor,
) -> Tensor:
    """ˆx = π(H [c0 + u_i + Δ_i, 1]) with center-relative coordinates."""
    rel = canonical_xy - center.unsqueeze(0) + local_delta
    lifted = rel + center.unsqueeze(0)
    return apply_homography_points(lifted, homography)
