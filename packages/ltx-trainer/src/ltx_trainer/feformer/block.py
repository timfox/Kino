"""Frequency-enhanced Transformer block (Sec. 3.2, Eq. 1–2)."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from ltx_trainer.feformer.fdsa import FDSA
from ltx_trainer.feformer.fgmlp import FGMLP


class FEFormerBlock(nn.Module):
    def __init__(self, channels: int, *, mlp_ratio: int = 4, dw_kernel: int = 7) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.fdsa = FDSA(channels, dw_kernel=dw_kernel)
        self.norm2 = nn.LayerNorm(channels)
        self.fgmlp = FGMLP(channels, mlp_ratio=mlp_ratio, dw_kernel=dw_kernel)

    def forward(self, x: Tensor) -> Tensor:
        b, c, d, h, w = x.shape
        n = self.norm1(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        x = self.fdsa(n) + x
        m = self.norm2(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return self.fgmlp(m) + x
