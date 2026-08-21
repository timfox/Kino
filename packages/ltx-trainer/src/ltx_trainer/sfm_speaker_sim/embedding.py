"""Speaker embedding extraction and cosine similarity (Sec. 3.2)."""

from __future__ import annotations

import numpy as np


def speaker_embedding(hidden_states: np.ndarray) -> np.ndarray:
    """Mean pool over frames: hidden_states shape (T, D)."""
    return hidden_states.mean(axis=0)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity for rows of embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    unit = embeddings / norms
    sim = unit @ unit.T
    np.fill_diagonal(sim, 0.0)
    return sim


def normalize_scores(scores: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map perceptual scores from [lo, hi] to [0, 1]."""
    return (scores - lo) / (hi - lo)
