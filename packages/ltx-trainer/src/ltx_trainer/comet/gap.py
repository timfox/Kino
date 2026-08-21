"""Modality gap sources and mitigation proxies."""

from __future__ import annotations

import numpy as np


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def embedding_shift(audio: np.ndarray, *, text_mean: np.ndarray, audio_mean: np.ndarray) -> np.ndarray:
    """Classic ES: subtract audio mean gap vector, add text mean."""
    gap = audio_mean - text_mean
    out = audio - gap
    n = np.linalg.norm(out)
    return out / (n + 1e-12)


def modality_gap_sources() -> list[dict[str, str]]:
    """Sec. V categorization of gap sources."""
    return [
        {"subspace": "mean", "description": "Static cone centroids t̄ vs ā (embedding shift target)"},
        {"subspace": "shared_head", "description": "Imperfect u_j·v_j alignment and coefficient mismatch"},
        {"subspace": "modality_tail", "description": "Unaligned private tail (~924 dims) with residual energy"},
    ]
