"""Camera trajectory and rotation (Sec. 3.3, Eq. 1–7)."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

Trajectory = Literal["egocentric", "non_egocentric"]


def camera_forward_from_positions(positions: Tensor) -> Tensor:
    """
    Horizontal forward d_proj from consecutive positions (Eq. 1).
    positions [N, 3] → unit vectors [N, 3] in X-Y plane.
    """
    n = positions.shape[0]
    if n < 2:
        return torch.tensor([0.0, -1.0, 0.0]).expand(n, 3)
    d = positions[:-1] - positions[1:]
    nz = torch.tensor([0.0, 0.0, 1.0], dtype=positions.dtype, device=positions.device)
    d_proj = d - nz * (d @ nz).unsqueeze(-1)
    d_proj = d_proj / d_proj.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.cat([d_proj, d_proj[-1:]], dim=0)


def rotation_world_to_camera(forward_xy: Tensor) -> Tensor:
    """
    Ri = Rx(θ) Rz(φ) (Eq. 5–7) simplified for stub trajectories.
    forward_xy: unit vector in X-Y plane [3].
    """
    fx, fy, _ = forward_xy.tolist()
    phi = math.atan2(-fx, -fy)
    cz, sz = math.cos(phi), math.sin(phi)
    rz = torch.tensor([[cz, sz, 0.0], [-sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=forward_xy.dtype)
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=forward_xy.dtype)
    return rx @ rz


def synthetic_egocentric_positions(n: int, radius: float = 2.0, height: float = 1.5) -> Tensor:
    """Spiral / circular ego trajectory (Sec. C.3)."""
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    x = -radius / 2 + radius * torch.cos(t)
    y = radius * torch.sin(t)
    z = height + 0.2 * torch.linspace(0, 1, n)
    return torch.stack([x, y, z], dim=-1)


def synthetic_non_egocentric_positions(n: int) -> Tensor:
    """Spline-like path through keypoints (Sec. C.3 stub)."""
    key = torch.tensor(
        [
            [0.0, 0.0, 1.2],
            [2.0, 1.0, 1.4],
            [3.5, -0.5, 1.6],
            [1.0, -2.0, 1.3],
            [-1.0, -1.0, 1.1],
        ]
    )
    t = torch.linspace(0, key.shape[0] - 1, n)
    idx0 = t.floor().long().clamp(0, key.shape[0] - 2)
    w = (t - idx0.float()).unsqueeze(-1)
    return key[idx0] * (1 - w) + key[idx0 + 1] * w
