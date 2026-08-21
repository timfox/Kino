"""First-Order Ambisonics helpers (Sec. 3.1, Appendix B.3)."""

from __future__ import annotations

import numpy as np

FOA_CHANNEL_NAMES = ("W", "X", "Y", "Z")


def pseudo_foa_from_stereo(left: np.ndarray, right: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """
    Curriculum pseudo-FOA: W = L+R, one axis carries L-R, others zero (Appendix B.3).
    Returns shape (4, L).
    """
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    n = min(left.size, right.size)
    left, right = left[:n], right[:n]
    w = left + right
    diff = left - right
    rng = np.random.default_rng(seed)
    axis = int(rng.integers(0, 3))  # X,Y,Z only
    foa = np.zeros((4, n), dtype=np.float64)
    foa[0] = w
    foa[1 + axis] = diff
    return foa


def intensity_vector_azimuth(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Toy DoA azimuth from W,X,Y cross-power (Eq. 5–6 stub on mean vectors)."""
    ix = float(np.mean(w * x))
    iy = float(np.mean(w * y))
    return float(np.arctan2(iy, ix))
