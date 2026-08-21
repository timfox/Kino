"""Panoramic Plücker embedding (Sec. 3.2, Eq. 5)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.campvg.erp_geometry import direction_grid


def panoramic_plucker_map(
    rotation: Tensor,
    translation: Tensor,
    height: int,
    width: int,
) -> Tensor:
    """
    P ∈ R^{H×W×6}: d = R·d̂, m = t × d.

    rotation: [3,3]; translation: [3].
    """
    dirs = direction_grid(height, width, device=rotation.device)
    d = torch.einsum("hwc,cd->hwd", dirs, rotation)
    d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    t = translation.view(1, 1, 3)
    m = torch.cross(t.expand_as(d), d, dim=-1)
    return torch.cat([m, d], dim=-1)


def trajectory_plucker(
    rotations: Tensor,
    translations: Tensor,
    height: int,
    width: int,
) -> Tensor:
    """[N, H, W, 6] for N frames."""
    n = rotations.shape[0]
    maps = [
        panoramic_plucker_map(rotations[i], translations[i], height, width)
        for i in range(n)
    ]
    return torch.stack(maps, dim=0)


class PanoramicPoseEncoder(nn.Module):
    """Linear pose encoder stub (CameraCtrl-style)."""

    def __init__(self, in_dim: int = 6, hidden: int = 32, out_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, out_dim, 3, padding=1),
        )

    def forward(self, plucker: Tensor) -> Tensor:
        """plucker [B, N, H, W, 6] → [B, N, C, H, W]."""
        b, n, h, w, c = plucker.shape
        x = plucker.permute(0, 1, 4, 2, 3).reshape(b * n, c, h, w)
        return self.proj(x).view(b, n, -1, h, w)
