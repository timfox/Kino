"""Embedding-based degradation detection toy (Sec. 5, arXiv:2605.21332)."""

from __future__ import annotations

import numpy as np


def cosine_similarity_to_enrollment(
    frame_embeddings: np.ndarray,
    enrollment: np.ndarray,
) -> np.ndarray:
    """Cosine similarity per frame: sim_l = cos(z_l, z_enroll) with z on unit sphere."""
    z = np.asarray(frame_embeddings, dtype=np.float64)
    e = np.asarray(enrollment, dtype=np.float64).ravel()
    if z.ndim != 2:
        raise ValueError("frame_embeddings must be (L, D)")
    if z.shape[1] != e.size:
        raise ValueError("enrollment dimension must match embedding dim")
    z_n = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    e_n = e / (np.linalg.norm(e) + 1e-8)
    return (z_n * e_n.reshape(1, -1)).sum(axis=1)


def detect_by_threshold(similarities: np.ndarray, threshold: float) -> np.ndarray:
    """Binary degraded-vs-clean decisions from cosine scores (higher = more like enrollment = clean proxy)."""
    s = np.asarray(similarities, dtype=np.float64).ravel()
    return (s < float(threshold)).astype(np.int64)
