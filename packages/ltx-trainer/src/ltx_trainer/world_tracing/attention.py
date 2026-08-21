"""Three-way attention factorization for WT-DiT."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PreNormAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        h = self.heads
        qkv = self.qkv(self.norm(x)).reshape(b, n, 3, h, d // h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return x + self.proj(out)


class WTDiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.layer_attn = PreNormAttention(dim, heads)
        self.ray_attn = PreNormAttention(dim, heads)
        self.global_attn = PreNormAttention(dim, heads)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        b, layers, patches, dim = tokens.shape
        x = tokens.reshape(b * layers, patches, dim)
        x = self.layer_attn(x).reshape(b, layers, patches, dim)
        x = x.permute(0, 2, 1, 3).reshape(b * patches, layers, dim)
        x = self.ray_attn(x).reshape(b, patches, layers, dim).permute(0, 2, 1, 3)
        x = x.reshape(b, layers * patches, dim)
        x = self.global_attn(x).reshape(b, layers, patches, dim)
        return x + self.ff(x)


class LayerFiLM(nn.Module):
    """Per-layer FiLM to break layer permutation symmetry."""

    def __init__(self, num_layers: int, dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_layers, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim * 2))

    def forward(self, tokens: Tensor) -> Tensor:
        b, layers, patches, dim = tokens.shape
        ids = torch.arange(layers, device=tokens.device)
        gamma_beta = self.mlp(self.embed(ids)).unsqueeze(0).unsqueeze(2)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma * tokens + beta


class AdaLNTime(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 2))

    def forward(self, tokens: Tensor, t_emb: Tensor) -> Tensor:
        while t_emb.ndim < tokens.ndim:
            t_emb = t_emb.unsqueeze(1)
        scale, shift = self.mlp(t_emb).chunk(2, dim=-1)
        return tokens * (1 + scale) + shift
