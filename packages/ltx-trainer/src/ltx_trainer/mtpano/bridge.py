"""Bridge Feature Extractor with truncated gradient flow (Sec. 3.2.2)."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class BridgeFeatureExtractor(nn.Module):
    """Cross-attention BFE: query=backbone, kv=concatenated task tokens (Zhang et al. 2025b)."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        *,
        detach_kv: bool = False,
    ) -> Tensor:
        """query, key_value: (B, N, C)."""
        kv = key_value.detach() if detach_kv else key_value
        return query + self.attn(query, kv, kv)[0]
