"""Equirectangular human track projection stub (Bacchin et al., Sec. III-B)."""

from __future__ import annotations

import torch
from torch import Tensor


def erp_pixel_to_robot_xy(
    u: Tensor,
    v: Tensor,
    *,
    width: int = 128,
    height: int = 64,
    ground_z: float = 0.0,
) -> Tensor:
    """Flat-ground stub: map ERP pixels to robot egocentric (x, y)."""
    lam = (u / width - 0.5) * 2.0 * 3.14159265
    depth = 1.0 + 2.0 * (0.5 - v / height).clamp(0.1, 1.0)
    x = depth * torch.sin(lam)
    y = depth * torch.cos(lam)
    _ = ground_z
    return torch.stack([x, y], dim=-1)


def tracks_from_keypoints_stub(num_agents: int = 3, steps: int = 6) -> Tensor:
    """Synthetic (B, A, T, 2) past trajectories in robot frame."""
    t = torch.linspace(0, 1, steps)
    base = torch.stack([t, 0.1 * torch.sin(t * 3)], dim=-1)
    agents = [base + torch.tensor([i * 0.5, 0.0]) for i in range(num_agents)]
    return torch.stack(agents, dim=0).unsqueeze(0)
