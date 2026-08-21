"""Slot query fusion and heatmap confidence (Sec. 3.1–3.2, Eq. 2, 5)."""

from __future__ import annotations

import math


def fuse_slot_query(concentration_emb: float, concept_repr: float) -> float:
    r"""Qk = MLP(Concat(ek, rk)) — scalar toy."""
    return 0.5 * (concentration_emb + concept_repr)


def attention_score(query: float, key: float, *, scale: float = 1.0) -> float:
    """Single-head dot-product score before map/conf heads."""
    return (query * key) / scale


def multi_head_attention_scores(
    queries: list[float],
    keys: list[float],
    *,
    num_heads: int = 8,
    head_dim: float = 64.0,
) -> list[list[float]]:
    r"""S = [(Q W^Q_i)(K W^K_i)^T / sqrt(d_h)] — per-head score matrix (Eq. 5)."""
    scale = math.sqrt(head_dim)
    scores: list[list[float]] = []
    for h in range(num_heads):
        head_scale = scale * (1.0 + 0.05 * h)
        row = [attention_score(q, k, scale=head_scale) for q in queries for k in keys]
        scores.append(row)
    return scores


def slot_confidence(attention_max: float) -> float:
    """Map attention peak to [0, 1] confidence."""
    return max(0.0, min(1.0, attention_max))
