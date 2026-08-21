"""Low-rank decomposition for D4C / offset nets (Sec. 3.3.3, Eq. 11)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LowRankConv2d(nn.Module):
    """Factorize W' ∈ R^{Ci×CoN²} as A B^T with rank r."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, rank: int = 8) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        flat_out = out_ch * kernel_size * kernel_size
        self.a = nn.Parameter(torch.randn(in_ch, rank) * 0.02)
        self.b = nn.Parameter(torch.randn(flat_out, rank) * 0.02)

    def weight_matrix(self) -> Tensor:
        w_flat = self.a @ self.b.t()  # Ci × CoN²
        n = self.kernel_size
        return w_flat.view(self.in_ch, self.out_ch, n, n).permute(1, 0, 2, 3)

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.conv2d(x, self.weight_matrix(), padding=self.kernel_size // 2)


def parameter_count_full_vs_lowrank(
    in_ch: int, out_ch: int, kernel_size: int = 3, rank: int = 8
) -> tuple[int, int]:
    full = in_ch * out_ch * kernel_size * kernel_size
    low = in_ch * rank + out_ch * kernel_size * kernel_size * rank
    return full, low
