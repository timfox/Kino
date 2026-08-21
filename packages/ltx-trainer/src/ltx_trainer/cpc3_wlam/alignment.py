"""Alignment-aware local and global acoustic fusion (toy)."""

from __future__ import annotations

import numpy as np


def sharpness_score(attn: np.ndarray) -> float:
    """Eq. (4) proxy for alignment-relevant head quality."""
    return float(np.linalg.norm(attn, axis=1).sum() + np.linalg.norm(attn, axis=0).sum())


def select_dynamic_topk_heads(attn_heads: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Select top-k heads by sharpness and return averaged map."""
    scores = np.array([sharpness_score(h) for h in attn_heads], dtype=np.float64)
    ids = np.argsort(scores)[-k:]
    return attn_heads[ids].mean(axis=0), ids


def word_aligned_local_summary(
    averaged_attn: np.ndarray,
    char_to_word: list[list[int]],
    encoder_states: np.ndarray,
) -> np.ndarray:
    """Eq. (5)-(6) toy: build per-word local acoustic summaries from attention."""
    out = np.zeros((len(char_to_word), encoder_states.shape[1]), dtype=encoder_states.dtype)
    for i, chars in enumerate(char_to_word):
        if not chars:
            continue
        alpha = averaged_attn[chars].mean(axis=0)
        alpha = alpha / (alpha.sum() + 1e-8)
        out[i] = alpha @ encoder_states
    return out


def utterance_global_summary(encoder_states: np.ndarray) -> np.ndarray:
    """Eq. (7) toy masked mean pooling."""
    return encoder_states.mean(axis=0)
