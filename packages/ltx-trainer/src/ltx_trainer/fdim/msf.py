"""Attention-based Multi-Scale Fusion (MSF, Eq. 8–11)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _CBAM(nn.Module):
    def __init__(self, ch: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(ch // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(ch, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, ch),
        )
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        x = x * w
        sm = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        x = x * torch.sigmoid(self.spatial(sm))
        return x


class MultiScaleFusion(nn.Module):
    def __init__(self, channels: tuple[int, int, int, int]) -> None:
        super().__init__()
        self.cbam = nn.ModuleList([_CBAM(c) for c in channels])
        self.total = sum(channels)

    def forward(self, feats: list[Tensor]) -> Tensor:
        vecs = []
        for i, h in enumerate(feats):
            refined = self.cbam[i](h)
            vecs.append(F.adaptive_avg_pool2d(refined, 1).flatten(1))
        return torch.cat(vecs, dim=1)
