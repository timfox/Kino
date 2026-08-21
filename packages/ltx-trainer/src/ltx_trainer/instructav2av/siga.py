"""Source-Instruction Gated Attention — SIGA (Eq. 5–6, Sec. 4.2)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
) -> Tensor:
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


class SIGAModule(nn.Module):
    """
    Noisy latent queries source + instruction; fuse via learned gate G.

    Eq. (5): f̂^m_h = Attn(Q, K^m, V^m) for m ∈ {x, c}
    Eq. (6): G = σ(W_g[f̂^x; f̂^c; f_h]),  f̃ = (1-G)⊙f̂^x + G⊙f̂^c
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk_src = nn.Linear(dim, dim, bias=False)
        self.wv_src = nn.Linear(dim, dim, bias=False)
        self.wk_inst = nn.Linear(dim, dim, bias=False)
        self.wv_inst = nn.Linear(dim, dim, bias=False)
        self.wg = nn.Linear(dim * 3, dim)

    def forward(
        self,
        f_h: Tensor,
        f_src: Tensor,
        f_inst: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Returns fused features and per-token gate mean."""
        q = self.wq(f_h)
        f_src_hat = scaled_dot_product_attention(q, self.wk_src(f_src), self.wv_src(f_src))
        f_inst_hat = scaled_dot_product_attention(q, self.wk_inst(f_inst), self.wv_inst(f_inst))
        gate_in = torch.cat([f_src_hat, f_inst_hat, f_h], dim=-1)
        g = torch.sigmoid(self.wg(gate_in))
        fused = (1.0 - g) * f_src_hat + g * f_inst_hat
        return fused, g.mean(dim=-1)
