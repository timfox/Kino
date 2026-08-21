"""Plücker ray embeddings for exploration routes (Sec. 3.3)."""

from __future__ import annotations

import torch
from torch import Tensor


def euler_to_rotation_matrix(yaw: Tensor, pitch: Tensor, roll: Tensor) -> Tensor:
    """R = Rz(yaw) Ry(pitch) Rx(roll) — stub for 6-DoF route."""
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)
    rz = torch.stack(
        [
            torch.stack([cy, -sy, torch.zeros_like(cy)], dim=-1),
            torch.stack([sy, cy, torch.zeros_like(cy)], dim=-1),
            torch.stack([torch.zeros_like(cy), torch.zeros_like(cy), torch.ones_like(cy)], dim=-1),
        ],
        dim=-2,
    )
    ry = torch.stack(
        [
            torch.stack([cp, torch.zeros_like(cp), sp], dim=-1),
            torch.stack([torch.zeros_like(cp), torch.ones_like(cp), torch.zeros_like(cp)], dim=-1),
            torch.stack([-sp, torch.zeros_like(cp), cp], dim=-1),
        ],
        dim=-2,
    )
    rx = torch.stack(
        [
            torch.stack([torch.ones_like(cr), torch.zeros_like(cr), torch.zeros_like(cr)], dim=-1),
            torch.stack([torch.zeros_like(cr), cr, -sr], dim=-1),
            torch.stack([torch.zeros_like(cr), sr, cr], dim=-1),
        ],
        dim=-2,
    )
    return rz @ ry @ rx


def plucker_embedding_stub(
    height: int,
    width: int,
    translation: Tensor,
    rotation: Tensor,
    *,
    focal: float = 500.0,
) -> Tensor:
    """
    Pixel-wise Plücker (d × (t × d), d) stub → [H, W, 6].

    translation: [3]; rotation: [3, 3].
    """
    device = translation.device
    dtype = translation.dtype
    v_coords = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    u_coords = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(v_coords, u_coords, indexing="ij")
    ones = torch.ones_like(uu)
    dirs_cam = torch.stack(
        [(uu - width / 2) / focal, (vv - height / 2) / focal, ones],
        dim=-1,
    )
    dirs_world = torch.einsum("hwc,cd->hwd", dirs_cam, rotation)
    dirs_world = dirs_world / dirs_world.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    moment = torch.cross(
        translation.expand_as(dirs_world),
        dirs_world,
        dim=-1,
    )
    return torch.cat([moment, dirs_world], dim=-1)
