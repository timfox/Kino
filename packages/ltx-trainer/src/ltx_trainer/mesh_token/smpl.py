"""SMPL motion decomposition: trajectory + canonical pose mesh (§3.2)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def canonicalize_vertices(vertices: Tensor) -> Tensor:
    """Translate to origin and remove global orientation (paper §3.2)."""
    centered = vertices - vertices.mean(dim=-2, keepdim=True)
    return centered


def pack_trajectory(
    gamma_world: Tensor,
    rot_world: Tensor,
    gamma_cam: Tensor,
    rot_cam: Tensor,
) -> Tensor:
    """Concat world/camera translation and flattened rotation (Eq. 3)."""
    rw = rot_world.reshape(*rot_world.shape[:-2], 9)
    rc = rot_cam.reshape(*rot_cam.shape[:-2], 9)
    return torch.cat([gamma_world, rw, gamma_cam, rc], dim=-1)


def smpl_motion_stub(batch: int = 1, frames: int = 25, vertices: int = 6890) -> dict[str, Tensor]:
    """Synthetic SMPL mesh sequence for smoke tests."""
    v = torch.randn(batch, frames, vertices, 3)
    return {
        "vertices": v,
        "canonical": canonicalize_vertices(v),
        "gamma_world": torch.randn(batch, frames, 3),
        "rot_world": torch.eye(3).expand(batch, frames, 3, 3).clone(),
        "gamma_cam": torch.randn(batch, frames, 3),
        "rot_cam": torch.eye(3).expand(batch, frames, 3, 3).clone(),
    }
