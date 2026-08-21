"""Toy contextual magnitude masking (Eq. 2, arXiv:2605.22262)."""

from __future__ import annotations

import numpy as np


def contextual_denoising_mask(
    magnitude_noisy: np.ndarray,
    context_embedding: np.ndarray,
    base: float = 0.75,
    span: float = 0.2,
) -> np.ndarray:
    """Toy mask in [0, 1]: stronger retention when context embedding has larger norm."""
    x = np.asarray(magnitude_noisy, dtype=np.float64)
    e = np.asarray(context_embedding, dtype=np.float64).ravel()
    gate = float(np.tanh(np.linalg.norm(e)))
    level = base + span * gate
    return np.full_like(x, min(max(level, 0.0), 1.0))


def masked_magnitude(
    magnitude_noisy: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """|X̂| = |X̃| ⊙ mask (Eq. 2)."""
    return np.asarray(magnitude_noisy, dtype=np.float64) * np.asarray(mask, dtype=np.float64)
