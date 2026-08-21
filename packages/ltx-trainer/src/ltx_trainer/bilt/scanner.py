"""BiLT cross-attention scanner encoder (Sec. 2.2.1, Table 1)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.bilt.config import BiLTConfig


class ImportanceGating(nn.Module):
    """Layer 5: mean/std/max statistics → MLP gate per spectral position."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, C)
        stats = torch.stack([x.mean(-1), x.std(-1), x.amax(-1)], dim=-1)
        gate = self.mlp(stats)  # (B, L, 1)
        return x * gate


class CrossAttentionBlock(nn.Module):
    def __init__(self, cfg: BiLTConfig) -> None:
        super().__init__()
        self.probes = nn.Parameter(torch.randn(cfg.num_probes, cfg.feature_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            cfg.feature_dim, cfg.num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(cfg.feature_dim)
        self.self_attn = nn.MultiheadAttention(
            cfg.feature_dim, cfg.num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(cfg.feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.ffn_dim),
            nn.GELU(),
            nn.Linear(cfg.ffn_dim, cfg.feature_dim),
        )
        self.norm3 = nn.LayerNorm(cfg.feature_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, C)
        b = x.shape[0]
        queries = self.probes.unsqueeze(0).expand(b, -1, -1)
        cross_out, _ = self.cross_attn(queries, x, x)
        cross_out = self.norm1(cross_out + queries)
        self_out, _ = self.self_attn(cross_out, cross_out, cross_out)
        cross_out = self.norm2(cross_out + self_out)
        ffn_out = self.ffn(cross_out)
        return self.norm3(cross_out + ffn_out)


class BiLTScanner(nn.Module):
    def __init__(self, cfg: BiLTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Conv1d(1, cfg.conv_filters, cfg.conv_kernel, padding=cfg.conv_kernel // 2)
        self.norm1 = nn.LayerNorm(cfg.conv_filters)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.LayerNorm(cfg.conv_filters)
        self.gating = ImportanceGating(cfg.conv_filters) if cfg.use_importance_gating else None
        self.pos_embed = nn.Embedding(cfg.spectral_points, cfg.conv_filters)
        self.attn = CrossAttentionBlock(cfg)
        self.bottleneck = nn.Sequential(
            nn.Linear(cfg.num_probes * cfg.feature_dim, cfg.bottleneck_dim),
            nn.GELU(),
        )
        self.latent = nn.Linear(cfg.bottleneck_dim, cfg.latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x: (B, L) normalized reflectance/transmittance
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = F.gelu(self.conv(x))  # (B, C, L)
        h = h.transpose(1, 2)  # (B, L, C)
        h = self.norm1(h)
        h = self.pool(h.transpose(1, 2)).transpose(1, 2)
        h = self.norm2(h)
        if self.gating is not None:
            h = self.gating(h)
        positions = torch.arange(h.shape[1], device=h.device)
        h = h + self.pos_embed(positions)
        probes = self.attn(h)  # (B, num_probes, C)
        flat = probes.flatten(1)
        z = F.softplus(self.latent(self.bottleneck(flat)))
        return z, probes
