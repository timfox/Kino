"""Scaled dot-product and bidirectional cross-attention (Eqs. 6–10)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Attn(Q,K,V) = softmax(QK^T / sqrt(d)) V."""
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


class CrossAttentionBlock(nn.Module):
    """Single-direction cross-attention with projections."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        b, t, _ = query.shape
        q = self.q_proj(query)
        k = self.k_proj(context)
        v = self.v_proj(context)
        h = scaled_dot_product_attention(q, k, v)
        h = self.out_proj(h)
        x = self.norm(query + h)
        return self.norm(x + self.ff(x))


class BidirectionalCrossFusion(nn.Module):
    """Audio→visual and visual→audio cross-attention (Eqs. 7–10)."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.a2v = CrossAttentionBlock(dim, num_heads)
        self.v2a = CrossAttentionBlock(dim, num_heads)

    def forward(self, visual: Tensor, audio: Tensor) -> tuple[Tensor, Tensor]:
        ha = self.a2v(audio, visual)
        hv = self.v2a(visual, audio)
        return ha, hv
