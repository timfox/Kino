"""Label-free P2E / E2P patch pseudo-label pipeline (Sec. 3.1, Eq. 5–6)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.mtpano.erp import spherical_ray_direction


def sample_perspective_pose(
    batch_size: int,
    *,
    device: torch.device | None = None,
    fov_range: tuple[float, float] = (80.0, 120.0),
) -> tuple[Tensor, Tensor, Tensor]:
    """Random yaw ψ, pitch η, FoV per patch."""
    dev = device or torch.device("cpu")
    yaw = torch.rand(batch_size, device=dev) * 360.0
    pitch = (torch.rand(batch_size, device=dev) - 0.5) * 180.0
    fov = fov_range[0] + torch.rand(batch_size, device=dev) * (fov_range[1] - fov_range[0])
    return yaw, pitch, fov


def yaw_pitch_rotation_matrix(yaw_deg: Tensor, pitch_deg: Tensor) -> Tensor:
    """R(η, ψ) for normal reprojection (Eq. 5–6), shape (B, 3, 3)."""
    yaw = yaw_deg * (math.pi / 180.0)
    pitch = pitch_deg * (math.pi / 180.0)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    rz = torch.stack(
        [
            torch.stack([cy, -sy, torch.zeros_like(cy)], dim=-1),
            torch.stack([sy, cy, torch.zeros_like(cy)], dim=-1),
            torch.stack([torch.zeros_like(cy), torch.zeros_like(cy), torch.ones_like(cy)], dim=-1),
        ],
        dim=-2,
    )
    rx = torch.stack(
        [
            torch.stack([torch.ones_like(cp), torch.zeros_like(cp), torch.zeros_like(cp)], dim=-1),
            torch.stack([torch.zeros_like(cp), cp, -sp], dim=-1),
            torch.stack([torch.zeros_like(cp), sp, cp], dim=-1),
        ],
        dim=-2,
    )
    return rz @ rx


def p2e_patch_stub(pano: Tensor, yaw_deg: Tensor, pitch_deg: Tensor) -> Tensor:
    """Perspective crop stub: yaw roll + half-res (Π_P2E stand-in for semseg RGB)."""
    b, _, _h, w = pano.shape
    shift = (yaw_deg / 360.0 * w).long() % w
    out = pano.clone()
    for i in range(b):
        out[i] = torch.roll(pano[i], shifts=-int(shift[i].item()), dims=-1)
    return F.interpolate(out, scale_factor=0.5, mode="bilinear", align_corners=False)


def e2p_depth(depth_persp: Tensor, cos_incidence: Tensor) -> Tensor:
    """Divide perspective depth by d_cam·k (Eq. 6, depth branch)."""
    return depth_persp / cos_incidence.clamp_min(1e-3)


def e2p_normals(normals_persp: Tensor, rotation: Tensor) -> Tensor:
    """Rotate normals to sphere frame: R · n (Eq. 6, normal branch)."""
    b, _, h, w = normals_persp.shape
    flat = normals_persp.permute(0, 2, 3, 1).reshape(b, h * w, 3)
    rotated = torch.bmm(flat, rotation.transpose(1, 2))
    return rotated.reshape(b, h, w, 3).permute(0, 3, 1, 2)


def metric_point_map_from_depth(depth: Tensor, phi: Tensor, theta: Tensor) -> Tensor:
    """P = D ⊙ r (supp. Eq. 8)."""
    r = spherical_ray_direction(phi, theta)
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    if r.dim() == 3:
        r = r.permute(2, 0, 1).unsqueeze(0)
    return depth * r
