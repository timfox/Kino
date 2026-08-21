"""Additive attention pooling over temporal sequences."""

from __future__ import annotations

import numpy as np


def attention_pool(
    frames: np.ndarray,
    mask: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Toy Bahdanau-style attention pooling (eq. 4)."""
    rng = np.random.default_rng(seed)
    wa = rng.standard_normal((frames.shape[1], frames.shape[1])) * 0.05
    w = rng.standard_normal(frames.shape[1]) * 0.05
    energies = np.tanh(frames @ wa) @ w
    if mask is not None:
        energies = np.where(mask > 0, energies, -1e9)
    alpha = np.exp(energies - energies.max())
    alpha = alpha / (alpha.sum() + 1e-8)
    return (alpha[:, None] * frames).sum(axis=0)
