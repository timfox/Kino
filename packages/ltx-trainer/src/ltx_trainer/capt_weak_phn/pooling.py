"""Pooling strategies for inducing phoneme-level predictors from higher-level labels (toy).

Key point from the paper:
- BASE: utterance score comes from a dedicated [CLS] head; without phoneme supervision, no phoneme head exists.
- MEAN/ATTN: utterance/word score is computed from pooled phoneme-level scores, so the phoneme head is trained
  even when only higher-level supervision is present.
"""

from __future__ import annotations

import numpy as np


def mean_pool(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return 0.0
    if mask is None:
        return float(v.mean())
    m = np.asarray(mask, dtype=np.float64)
    denom = float(m.sum())
    if denom <= 0:
        return 0.0
    return float((v * m).sum() / denom)


def attention_weights(hidden: np.ndarray, seed: int = 0) -> np.ndarray:
    """Toy attention head producing non-negative weights that sum to 1."""
    h = np.asarray(hidden, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError("hidden must be (T,D)")
    if h.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((h.shape[1],), dtype=np.float64) * 0.1
    logits = h @ w
    logits = logits - float(logits.max())
    ex = np.exp(logits)
    return ex / (float(ex.sum()) + 1e-12)


def attn_pool(values: np.ndarray, hidden: np.ndarray, mask: np.ndarray | None = None, seed: int = 0) -> float:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return 0.0
    a = attention_weights(hidden, seed=seed)
    if mask is not None:
        m = np.asarray(mask, dtype=np.float64)
        a = a * m
        a = a / (float(a.sum()) + 1e-12)
    return float((a * v).sum())


def pool_over_spans(
    phoneme_scores: np.ndarray,
    spans: list[list[int]],
    hidden_states: np.ndarray | None = None,
    mode: str = "MEAN",
    seed: int = 0,
) -> np.ndarray:
    """Pool phoneme scores per span (word or utterance units).

    Args:
        phoneme_scores: (P,)
        spans: list of phoneme indices for each unit
        hidden_states: (P, D) used for ATTN pooling
    """
    p = np.asarray(phoneme_scores, dtype=np.float64)
    out = np.zeros((len(spans),), dtype=np.float64)
    for i, ids in enumerate(spans):
        if not ids:
            continue
        vals = p[ids]
        if mode.upper() == "MEAN":
            out[i] = float(vals.mean())
        elif mode.upper() == "ATTN":
            if hidden_states is None:
                raise ValueError("hidden_states required for ATTN pooling")
            out[i] = attn_pool(vals, hidden_states[ids], seed=seed)
        else:
            raise ValueError(f"unknown pooling mode: {mode}")
    return out
