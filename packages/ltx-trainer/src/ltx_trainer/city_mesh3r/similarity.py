"""Global descriptor similarity graph (Sec. 3.1)."""

from __future__ import annotations

import torch
from torch import Tensor


def cosine_similarity_matrix(features: Tensor) -> Tensor:
    """Pairwise cosine similarity for L2-normalized rows."""
    if features.ndim != 2:
        raise ValueError("features must be (N, d)")
    normed = torch.nn.functional.normalize(features, dim=-1)
    return normed @ normed.T


def build_similarity_graph(
    features: Tensor,
    *,
    threshold: float,
) -> Tensor:
    """Return adjacency mask (N, N) with self-loops."""
    sim = cosine_similarity_matrix(features)
    adj = sim >= threshold
    adj.fill_diagonal_(True)
    return adj


def dinov2_feature_smoke(num_images: int, dim: int = 64, *, seed: int = 0) -> Tensor:
    """Toy global descriptors Φ(I) ∈ R^d."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(num_images, dim, generator=g)
