"""Parameter-free one-class fraud detector (Eq. 9)."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def sigmoid(x: Array | float) -> Array | float:
    x_arr = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x_arr))


def fraud_scores(h_final: Array) -> Array:
    """Eq. (9): s_i = σ(||h_final_i||_2)."""
    norms = np.linalg.norm(h_final, axis=1)
    return sigmoid(norms)
