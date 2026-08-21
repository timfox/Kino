"""Cross-domain attention (CDAM), Eq. (5)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class CrossDomainAttention(nn.Module):
    """Q from spatio-temporal path; K, V from frequency path (Fig. 3)."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q_tokens: Tensor, kv_tokens: Tensor) -> Tensor:
        """q_tokens, kv_tokens: (B, N, D)."""
        b, n, _ = q_tokens.shape
        q = self._heads(self.q_proj(q_tokens))
        k = self._heads(self.k_proj(kv_tokens))
        v = self._heads(self.v_proj(kv_tokens))
        scale = math.sqrt(self.head_dim)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / scale, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, self.dim)
        out = self.out_proj(out)
        fused = self.norm1(q_tokens + out)
        ffn_out = self.ffn(fused)
        return self.norm2(fused + ffn_out)

    def _heads(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
