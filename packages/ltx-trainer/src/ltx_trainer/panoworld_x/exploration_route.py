"""6-DoF exploration route ErTo = (x, y, z, α, β, γ) (Sec. 3.3)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.panoworld_x.plucker import euler_to_rotation_matrix, plucker_embedding_stub


class ExplorationRouteEncoder(nn.Module):
    """3D conv route encoder → latent-shaped condition (ControlNet-style stub)."""

    def __init__(self, in_channels: int = 6, hidden: int = 32, out_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, route: Tensor) -> Tensor:
        """route: [B, T, H, W, 6] → [B, C, T', H', W']."""
        x = route.permute(0, 4, 1, 2, 3)
        return self.net(x)


def route_to_plucker_volume(
    poses: Tensor,
    height: int,
    width: int,
) -> Tensor:
    """
    poses: [T, 6] with (x,y,z, yaw, pitch, roll) per frame.
    Returns [T, H, W, 6].
    """
    t = poses.shape[0]
    out = []
    for i in range(t):
        tr = poses[i, :3]
        r = euler_to_rotation_matrix(poses[i, 3], poses[i, 4], poses[i, 5])
        out.append(plucker_embedding_stub(height, width, tr, r))
    return torch.stack(out, dim=0)
