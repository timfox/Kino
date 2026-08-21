"""Bi-Stream Teacher Cosine Alignment — BiT-C (paper Eq. 3–5)."""

from __future__ import annotations

import numpy as np


def temporal_pool_layer_norm(h: np.ndarray) -> np.ndarray:
    """Mean-pool over time then L2-normalize (Eq. 3 stub)."""
    if h.ndim == 3:
        pooled = h.mean(axis=1)
    else:
        pooled = h.reshape(h.shape[0], -1)
    norms = np.linalg.norm(pooled, axis=-1, keepdims=True) + 1e-9
    return pooled / norms


def cosine_alignment(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def bitc_score(projected: np.ndarray, teacher: np.ndarray) -> float:
    """BiT-C cosine similarity (Eq. 4)."""
    return cosine_alignment(projected, teacher)


def bitc_loss(projected: np.ndarray, teacher: np.ndarray) -> float:
    """1 − BiT-C for interface distillation (Eq. 5)."""
    return 1.0 - bitc_score(projected, teacher)


def dual_bitc_loss(
    speech_proj: np.ndarray,
    speech_teacher: np.ndarray,
    audio_proj: np.ndarray,
    audio_teacher: np.ndarray,
) -> float:
    return bitc_loss(speech_proj, speech_teacher) + bitc_loss(audio_proj, audio_teacher)
