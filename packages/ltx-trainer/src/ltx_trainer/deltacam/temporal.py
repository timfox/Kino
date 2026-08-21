"""Lightweight temporal aggregation for camera-style embeddings (Fig. 6).

Stands in for the paper's temporal transformer over c-RADIO patch features. Input is
per-frame feature rows ``[B, T, D]``; output is time-aligned embeddings for style/content heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class TemporalStyleEncoder(nn.Module):
    """Shared temporal encoder before content/style bifurcation (Sec. 3.3)."""

    def __init__(
        self,
        d_model: int,
        *,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: Tensor) -> Tensor:
        """``x``: ``[B, T, D]`` → same shape."""
        if x.dim() != 3:
            raise ValueError(f"Expected [B, T, D], got {tuple(x.shape)}")
        return self.encoder(x)
