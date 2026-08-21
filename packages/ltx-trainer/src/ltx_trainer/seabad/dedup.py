"""Acoustic deduplication embedding toy (§3.1.3)."""

from __future__ import annotations

import numpy as np


def mel_embedding_mean_std(mel: np.ndarray) -> np.ndarray:
    """z = [μ(E), σ(E)] over time frames, shape (2 * n_mels,)."""
    if mel.ndim != 2:
        raise ValueError("mel must be (n_mels, n_frames)")
    mu = mel.mean(axis=1)
    sigma = mel.std(axis=1)
    z = np.concatenate([mu, sigma]).astype(np.float64)
    norm = np.linalg.norm(z)
    if norm > 0.0:
        z /= norm
    return z


def is_exact_duplicate(zi: np.ndarray, zj: np.ndarray, *, l2_threshold: float = 1e-7) -> bool:
    """Conservative duplicate criterion: ||zi − zj||₂ < threshold."""
    return float(np.linalg.norm(zi - zj)) < l2_threshold
