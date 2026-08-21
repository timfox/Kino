"""LoRA as parametric memory probe (Eq. 5)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def lora_forward(x: Tensor, w0: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """h = W0 x + B A x."""
    return torch.nn.functional.linear(x, w0) + (x @ a.T) @ b.T


def effective_lora_params(rank: int, d_in: int, d_out: int) -> int:
    """Trainable count for one adapted layer."""
    return rank * (d_in + d_out)


class LoRALinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, rank: int) -> None:
        super().__init__()
        self.w0 = nn.Linear(d_in, d_out, bias=False)
        self.a = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.b = nn.Parameter(torch.zeros(d_out, rank))

    def forward(self, x: Tensor) -> Tensor:
        return lora_forward(x, self.w0.weight, self.a, self.b)
