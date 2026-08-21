"""Spherical Geodesic-based Implicit Image Function (Eq. 9)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SGIF(nn.Module):
    """yp = SGIF(bzp, [θ, φ])."""

    def __init__(self, latent_dim: int, coord_dim: int = 2) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + coord_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 3),
        )

    def forward(self, z: Tensor, coords: Tensor) -> Tensor:
        b, c, h, w = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, c)
        if coords.shape[0] == 1 and coords.shape[1] == h * w:
            coord_flat = coords.squeeze(0)
        else:
            ys = torch.linspace(-1, 1, h, device=z.device, dtype=z.dtype)
            xs = torch.linspace(-1, 1, w, device=z.device, dtype=z.dtype)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            coord_flat = torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=-1)
        inp = torch.cat([z_flat, coord_flat[: z_flat.shape[0]]], dim=-1)
        rgb = self.mlp(inp)
        return rgb.view(b, h, w, 3).permute(0, 3, 1, 2)
