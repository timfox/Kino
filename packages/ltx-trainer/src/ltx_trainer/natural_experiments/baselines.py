"""Non-causal and classical causal feature-selection baselines (smoke rankings)."""

from __future__ import annotations

from typing import Any

import numpy as np


def pearson_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    scores = np.zeros(x.shape[1], dtype=np.float64)
    for j in range(x.shape[1]):
        c = np.corrcoef(x[:, j], y)[0, 1]
        scores[j] = abs(c) if np.isfinite(c) else 0.0
    return scores


def select_top_k(scores: np.ndarray, k: int) -> list[int]:
    k = max(1, min(k, len(scores)))
    return sorted(np.argsort(scores)[::-1][:k].tolist())


def baseline_feature_sets(
    x: np.ndarray,
    y: np.ndarray,
    *,
    mb_size: int = 4,
) -> dict[str, list[int]]:
    """Return top-k feature indices per baseline (§5)."""
    scores = pearson_scores(x, y)
    spearman = scores  # smoke: same ranking
    mi = scores * 1.1  # smoke proxy
    return {
        "pearson": select_top_k(scores, mb_size),
        "spearman": select_top_k(spearman, mb_size),
        "mi": select_top_k(mi, mb_size),
        "anova": select_top_k(scores, mb_size),
        "chi2": select_top_k(scores, mb_size),
        "boruta": select_top_k(scores, max(mb_size, int(0.6 * x.shape[1]))),
    }


def baselines_smoke() -> dict[str, Any]:
    rng = np.random.default_rng(51)
    x = rng.normal(size=(200, 11)).astype(np.float32)
    y = rng.integers(0, 3, size=200)
    return baseline_feature_sets(x, y, mb_size=4)
