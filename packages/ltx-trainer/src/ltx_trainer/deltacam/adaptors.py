"""Lightweight adaptors for style transfer (stage 2) and EXIF bridge (Fig. 7)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class StyleToDeltaAdaptor(nn.Module):
    """Map reference style embedding (+ optional EXIF ξ) into Δ-intrinsic control space."""

    def __init__(self, style_dim: int, num_intrinsics: int, *, exif_dim: int = 0, hidden: int = 256) -> None:
        super().__init__()
        in_dim = style_dim + exif_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_intrinsics),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z_style: Tensor, xi: Tensor | None = None) -> Tensor:
        if z_style.dim() == 1:
            z_style = z_style.unsqueeze(0)
        if xi is not None:
            if xi.dim() == 1:
                xi = xi.unsqueeze(0)
            if xi.shape[0] != z_style.shape[0]:
                raise ValueError("z_style and xi batch mismatch")
            x = torch.cat([z_style, xi], dim=-1)
        else:
            x = z_style
        return self.mlp(x)
