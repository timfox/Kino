"""Layerwise Attention Pooling (LAP) for Squeeze-MLLM (arXiv:2605.26111)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LayerwiseAttentionPooling(nn.Module):
    """Aggregate MLLM layer stack along the layer dimension (Fig. 2b).

    Input ``features`` has shape ``[B, M, L, C]`` where M is the number of MLLM layers,
    L is token sequence length, and C is channel width. Output is ``[B, L, C]``.
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.proj = nn.Linear(channels, 1)

    def forward(self, features: Tensor) -> Tensor:
        if features.dim() != 4:
            raise ValueError(f"features must be [B,M,L,C], got {tuple(features.shape)}")
        b, m, seq_len, c = features.shape
        # Attention over layer axis M for each (batch, token) pair.
        x = features.permute(0, 2, 1, 3).reshape(b * seq_len, m, c)
        attn_out, _ = self.attn(x, x, x)
        weights = torch.softmax(self.proj(attn_out).squeeze(-1), dim=-1)  # [B*L, M]
        pooled = torch.sum(attn_out * weights.unsqueeze(-1), dim=1)  # [B*L, C]
        return pooled.view(b, seq_len, c)
