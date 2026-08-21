"""Lightweight causal encoder for simplex sequences."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.cast.config import CASTConfig


class SimplexEmbedding(nn.Module):
    def __init__(self, cfg: CASTConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(cfg.support_dim, cfg.hidden_dim)

    def forward(self, p: Tensor) -> Tensor:
        return self.proj(p)


class CausalTransformerEncoder(nn.Module):
    """Decoder-only causal Transformer stub over simplex embeddings."""

    def __init__(self, cfg: CASTConfig) -> None:
        super().__init__()
        self.embed = SimplexEmbedding(cfg)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)

    def forward(self, seq: Tensor) -> Tensor:
        """seq: (B, T, D) → hidden (B, T, H)."""
        h = self.embed(seq)
        t = seq.shape[1]
        mask = torch.triu(torch.ones(t, t, device=seq.device, dtype=torch.bool), diagonal=1)
        return self.encoder(h, mask=mask)
