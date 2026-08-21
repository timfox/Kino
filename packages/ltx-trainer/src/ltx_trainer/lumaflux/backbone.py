"""Frozen MM-DiT backbone stub (Sec. 3.1, 4)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MMDiTBlockStub(nn.Module):
    """Single Luma-MMDiT block with frozen backbone + adapter hooks."""

    def __init__(self, dim: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.num_heads = num_heads
        for p in self.qkv.parameters():
            p.requires_grad = False
        for p in self.proj.parameters():
            p.requires_grad = False
        for p in self.mlp.parameters():
            p.requires_grad = False

    def _attn(self, x: Tensor, v_override: Tensor | None = None) -> Tensor:
        b, n, d = x.shape
        qkv = self.qkv(self.ln1(x)).reshape(b, n, 3, self.num_heads, d // self.num_heads)
        q, k, v = qkv.unbind(dim=2)
        if v_override is not None:
            v = v_override.reshape(b, n, self.num_heads, -1)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (d // self.num_heads) ** 0.5, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj(out)

    def forward(self, x: Tensor, v_override: Tensor | None = None, h_override: Tensor | None = None) -> Tensor:
        h = self._attn(x, v_override=v_override)
        if h_override is not None:
            h = h_override
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x
