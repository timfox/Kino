"""SWG-Transformer stub (LGT-Net [10], Sec. 3.3.2)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class WindowAttentionStub(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: B x C x H x W → pool windows along H
        b, c, h, w = x.shape
        pooled = x.mean(dim=2)
        out = self.mlp(pooled.transpose(1, 2)).transpose(1, 2)
        return out.unsqueeze(2).expand(-1, -1, h, -1)


class SWGTransformer(nn.Module):
    """Window → Global → Shifted Window → Global (× repeats)."""

    def __init__(self, dim: int, repeats: int = 2) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        for _ in range(repeats):
            blocks.extend(
                [
                    WindowAttentionStub(dim),
                    WindowAttentionStub(dim),
                    WindowAttentionStub(dim),
                    WindowAttentionStub(dim),
                ]
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = x + 0.1 * blk(x)
        return x
