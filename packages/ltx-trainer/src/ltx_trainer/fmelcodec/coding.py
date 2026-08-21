"""Mel-spectrogram coding stage losses (Eq. 8–10)."""

from __future__ import annotations

import numpy as np


def mel_reconstruction_loss(natural: np.ndarray, coarse: np.ndarray) -> float:
    """Eq. (8): L1 + L2 mel reconstruction."""
    diff = natural - coarse
    return float(np.abs(diff).sum() + np.square(diff).sum())


def vq_loss(z: np.ndarray, z_q: np.ndarray, *, eta: float = 4.0) -> float:
    """Eq. (9): VQ-VAE codebook + commitment (stop-gradient via detached copies)."""
    z = np.asarray(z, dtype=np.float64)
    z_q = np.asarray(z_q, dtype=np.float64)
    codebook_term = float(np.sum((z_q - z) ** 2))
    commit_term = float(np.sum((z - z_q) ** 2))
    return codebook_term + eta * commit_term


def coding_stage_loss(
    natural: np.ndarray,
    coarse: np.ndarray,
    z: np.ndarray,
    z_q: np.ndarray,
    *,
    lambda_mel_rec: float = 45.0,
    lambda_vq: float = 2.5,
    eta: float = 4.0,
) -> float:
    """Eq. (10): L_cod = λ_mel L_mel-rec + λ_vq L_vq."""
    l_mel = mel_reconstruction_loss(natural, coarse)
    l_vq = vq_loss(z, z_q, eta=eta)
    return lambda_mel_rec * l_mel + lambda_vq * l_vq
