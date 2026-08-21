"""Hidden-state response trigger (Sec. 4.3, Eq. 5)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ResponseTrigger(nn.Module):
    """Cross-attention trigger: Respond vs Wait from prefix hidden states."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.cross = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 2),
        )

    def forward(self, hidden_prefix: Tensor) -> Tensor:
        """hidden_prefix (B, K, D) -> logits (B, 2)."""
        b = hidden_prefix.shape[0]
        q = self.query.expand(b, -1, -1)
        z, _ = self.cross(q, hidden_prefix, hidden_prefix)
        return self.head(z.squeeze(1))

    def predict(self, hidden_prefix: Tensor) -> Tensor:
        """Return 0=Wait, 1=Respond."""
        return self.forward(hidden_prefix).argmax(dim=-1)
