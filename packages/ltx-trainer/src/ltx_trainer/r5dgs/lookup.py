"""Offline CLIP-style object lookup (Sec. II-D, Eq. 8) — embeddings only, no model weights."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ObjectEmbeddingLookup(nn.Module):
    """Stores normalized per-group embeddings ``t_g``; retrieves by cosine similarity."""

    def __init__(self, num_groups: int, embed_dim: int) -> None:
        super().__init__()
        self.table = nn.Parameter(torch.randn(num_groups, embed_dim))

    def forward(self) -> Tensor:
        return F.normalize(self.table, dim=-1)

    def retrieve_group(self, text_embedding: Tensor) -> int:
        """Eq. (8): argmax cosine; ``text_embedding`` (D,) or (1, D)."""
        t = self.forward()
        q = F.normalize(text_embedding.reshape(-1, text_embedding.shape[-1]), dim=-1)
        sim = (t * q).sum(dim=-1)
        return int(sim.argmax().item())

    def retrieve_group_soft(self, text_embedding: Tensor) -> Tensor:
        """Cosine similarities (num_groups,)."""
        t = self.forward()
        q = F.normalize(text_embedding.reshape(-1, text_embedding.shape[-1]), dim=-1)
        return (t * q).sum(dim=-1)
