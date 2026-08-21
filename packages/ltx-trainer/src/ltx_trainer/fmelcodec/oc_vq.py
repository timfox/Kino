"""OC-VQ: single-codebook VQ with online clustering (Eq. 2–7, 4)."""

from __future__ import annotations

import math

import numpy as np


def quantize_nearest(z: np.ndarray, codebook: np.ndarray) -> tuple[int, np.ndarray]:
    """Eq. (2–3): nearest codevector assignment."""
    z = np.asarray(z, dtype=np.float64)
    codebook = np.asarray(codebook, dtype=np.float64)
    dists = np.sum((codebook - z) ** 2, axis=1)
    idx = int(np.argmin(dists))
    return idx, codebook[idx].copy()


def bitrate_bps(
    *,
    sample_rate: int = 16000,
    temporal_downsample: int = 4,
    frame_shift: int = 160,
    codebook_size: int = 1024,
) -> float:
    """Eq. (4): Bitrate = fs / (r * ws) * log2(K)."""
    return sample_rate / (temporal_downsample * frame_shift) * math.log2(codebook_size)


def ema_usage(
    prev_pi: float,
    batch_count: int,
    batch_size: int,
    *,
    rho: float = 0.999,
) -> float:
    """Eq. (5): EMA codevector usage rate."""
    return rho * prev_pi + (1.0 - rho) * batch_count / batch_size


def refresh_coefficient(pi_k: float, *, codebook_size: int = 1024, rho: float = 0.999, delta: float = 1e-3) -> float:
    """Eq. (6): γ_k for underused codevectors."""
    return math.exp(-10.0 * pi_k * codebook_size / (1.0 - rho) - delta)


def update_codevector(
    w_prev: np.ndarray,
    anchor: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Eq. (7): online clustering codevector refresh."""
    return (1.0 - gamma) * w_prev + gamma * anchor
