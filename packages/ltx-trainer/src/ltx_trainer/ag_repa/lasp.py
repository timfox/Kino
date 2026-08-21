"""Layer-wise Analysis via Shared Projection — LASP (paper Eq. 6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.ag_repa.bitc import bitc_score, temporal_pool_layer_norm


@dataclass(frozen=True)
class LayerScore:
    layer: int
    score: float


def lasp_layer_score(
    hidden: np.ndarray,
    *,
    projection: np.ndarray,
    teacher: np.ndarray,
) -> float:
    """Cosine similarity in shared teacher space after frozen projection."""
    pooled = temporal_pool_layer_norm(hidden)
    if projection.ndim == 2:
        projected = pooled @ projection.T
    else:
        projected = pooled * projection
    return bitc_score(projected, teacher)


def lasp_profile(
    layer_hiddens: dict[int, np.ndarray],
    *,
    projection: np.ndarray,
    teacher: np.ndarray,
) -> list[LayerScore]:
    scores = [
        LayerScore(layer=l, score=lasp_layer_score(h, projection=projection, teacher=teacher))
        for l, h in sorted(layer_hiddens.items())
    ]
    return scores


def top_k_layers(scores: list[LayerScore], k: int) -> list[int]:
    ranked = sorted(scores, key=lambda s: s.score, reverse=True)
    return [s.layer for s in ranked[:k]]
