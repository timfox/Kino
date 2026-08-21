"""Cross-attention fusion stub (Eq. 4)."""

from __future__ import annotations

import numpy as np


def cross_attention(
    acoustic: np.ndarray,
    linguistic: np.ndarray,
) -> np.ndarray:
    """C = CrossAttn(A, L, L) with scaled dot-product attention."""
    if linguistic.size == 0 or acoustic.size == 0:
        return acoustic
    q = acoustic
    k = linguistic
    v = linguistic
    scale = np.sqrt(q.shape[-1])
    scores = (q @ k.T) / scale
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True) + 1e-8
    return weights @ v


def fuse_for_prediction(acoustic: np.ndarray, context: np.ndarray) -> np.ndarray:
    """Concatenate acoustic frame features with attended linguistic context."""
    if context.ndim == 1:
        context = context.reshape(1, -1)
    if acoustic.ndim == 1:
        acoustic = acoustic.reshape(1, -1)
    t = min(acoustic.shape[0], context.shape[0])
    return np.concatenate([acoustic[:t], context[:t]], axis=-1)
