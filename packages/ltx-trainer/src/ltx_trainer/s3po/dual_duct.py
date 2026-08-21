"""Dual-duct residual refinement (Eq. 4–6, Fig. 7)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _DuctRefine(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DualDuctBlock(nn.Module):
    """Mutual exchange between Glocal and Gglobal (Eq. 4)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.r_local = _DuctRefine(dim)
        self.r_global = _DuctRefine(dim)

    def forward(self, glocal: Tensor, gglobal: Tensor) -> tuple[Tensor, Tensor]:
        rl, rg = self.r_local(glocal), self.r_global(gglobal)
        ff_local = glocal + rl + rg
        ff_global = gglobal + rg + rl
        return ff_local, ff_global


class DualDuctStack(nn.Module):
    def __init__(self, dim: int, num_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([DualDuctBlock(dim) for _ in range(num_blocks)])
        self.head_local = nn.Conv2d(dim, dim, 3, padding=1)
        self.head_global = nn.Conv2d(dim, dim, 3, padding=1)
        self.hidden = nn.Conv2d(dim * 2, dim, 3, padding=1)

    def forward(
        self, glocal: Tensor, gglobal: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        loc, glob = glocal, gglobal
        for blk in self.blocks:
            loc, glob = blk(loc, glob)
        lf = self.head_local(loc)
        gf = self.head_global(glob)
        h = F.relu(self.hidden(torch.cat([F.relu(lf), F.relu(gf)], dim=1)))
        return lf, gf, h
