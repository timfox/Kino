"""Rank-32 LoRA adapter stub for DiT blocks (Sec. 3.3)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 32) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_f, out_f = base.in_features, base.out_features
        self.a = nn.Linear(in_f, rank, bias=False)
        self.b = nn.Linear(rank, out_f, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.b(self.a(x))
