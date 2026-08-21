"""Inflated self/cross-attention across T=6 faces (Sec. 4.1)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class InflatedSelfAttention(nn.Module):
    """Tokens shaped B × (T·HW) × C with full cubemap self-attention."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        b, t, c, h, w = x.shape
        seq = x.reshape(b, t * h * w, c)
        out, _ = self.attn(seq, seq, seq)
        return out.reshape(b, t, h, w, c).permute(0, 1, 4, 2, 3)


class InflatedCrossAttention(nn.Module):
    def __init__(self, dim: int, ctx_dim: int, heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, kdim=ctx_dim, vdim=ctx_dim, batch_first=True)

    def forward(self, x: Tensor, ctx: Tensor) -> Tensor:
        b, t, c, h, w = x.shape
        seq = x.reshape(b, t * h * w, c)
        if ctx.ndim == 2:
            ctx = ctx.unsqueeze(1).expand(-1, seq.shape[1], -1)
        out, _ = self.attn(seq, ctx, ctx)
        return out.reshape(b, t, h, w, c).permute(0, 1, 4, 2, 3)
