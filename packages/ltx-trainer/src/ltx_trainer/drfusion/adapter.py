"""Condition Adapter P_str for structure-guided fusion (Eq. 3)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ConditionAdapter(nn.Module):
    """Extract c_struct from IR and inject into 3D-DiT input (zero-init convs)."""

    def __init__(self, in_channels: int = 1, latent_channels: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, latent_channels, 3, padding=1),
        )
        # Zero-init last layer for null structural signal at start
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, ir: Tensor) -> Tensor:
        """c_struct = P_str(I_IR)."""
        return self.encoder(ir)

    def fuse_input(self, z_vis: Tensor, c_struct: Tensor, pos: Tensor | None = None) -> Tensor:
        """X_in = Embed(z_VI) + c_struct + PosEmbed (Eq. 3)."""
        if c_struct.shape[-2:] != z_vis.shape[-2:]:
            c_struct = torch.nn.functional.interpolate(
                c_struct, size=z_vis.shape[-2:], mode="bilinear", align_corners=False
            )
        out = z_vis + c_struct
        if pos is not None:
            out = out + pos
        return out
