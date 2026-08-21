"""Activation preconditioning (AP) for DoPr (Algorithm 1–2)."""

from __future__ import annotations

import numpy as np


def uncentered_covariance(activations: np.ndarray) -> np.ndarray:
    """Empirical uncentered Σ_z = E[zz^T] with activations shape (n, d)."""
    if activations.ndim != 2:
        raise ValueError("activations must be (batch, dim)")
    n = activations.shape[0]
    if n == 0:
        raise ValueError("empty batch")
    return (activations.T @ activations) / float(n)


def damped_covariance(
    sigma: np.ndarray,
    *,
    damping: float,
    mode: str = "trace",
) -> np.ndarray:
    d = sigma.shape[0]
    if mode == "trace":
        gamma = damping * float(np.trace(sigma)) / float(d)
    elif mode == "fixed":
        gamma = damping
    else:
        raise ValueError(f"unknown damping mode: {mode}")
    return sigma + gamma * np.eye(d, dtype=sigma.dtype)


def activation_precondition(
    gradient: np.ndarray,
    activations: np.ndarray,
    *,
    damping: float = 1e-4,
    damping_mode: str = "trace",
) -> np.ndarray:
    """AP: M = G Σ_z^{-1} for weight gradient G (d_out, d_in) and inputs z (n, d_in)."""
    if gradient.ndim != 2:
        raise ValueError("gradient must be (d_out, d_in)")
    sigma = uncentered_covariance(activations)
    a = damped_covariance(sigma, damping=damping, mode=damping_mode)
    inv = np.linalg.inv(a)
    return gradient @ inv


def embedding_ap(gradient: np.ndarray, token_counts: np.ndarray) -> np.ndarray:
    """Efficient AP for one-hot embeddings: Σ is diagonal token counts."""
    counts = np.asarray(token_counts, dtype=np.float64)
    d = counts.shape[0]
    gamma = 1e-8 * float(np.sum(counts)) / max(d, 1)
    inv_diag = np.zeros_like(counts)
    seen = counts > 0.0
    inv_diag[seen] = 1.0 / (counts[seen] + gamma)
    return gradient * inv_diag[np.newaxis, :]
