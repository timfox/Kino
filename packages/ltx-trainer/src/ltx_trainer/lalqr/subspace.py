"""Reduced-order contrastive subspace via randomized SVD sketch (Sec. 4.2)."""

from __future__ import annotations

import numpy as np


def contrastive_rows(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """Rows are x(p+) - x(p-) for paired prompts — shape (N, D)."""
    if positive.shape != negative.shape:
        raise ValueError("positive and negative activations must share shape")
    return positive - negative


def mean_contrastive_vector(rows: np.ndarray) -> np.ndarray:
    """μ from Eq. (13)."""
    return rows.mean(axis=0)


def captured_energy_fraction(basis: np.ndarray, mu: np.ndarray) -> float:
    """ρ from Eq. (14): ||V^T μ||^2 / ||μ||^2."""
    mu = np.asarray(mu, dtype=np.float64)
    denom = float(np.dot(mu, mu))
    if denom <= 1e-12:
        return 0.0
    proj = basis.T @ mu
    return float(np.dot(proj, proj) / denom)


def randomized_svd_basis(
    rows: np.ndarray,
    *,
    rank: int,
    oversampling: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Orthonormal V ∈ R^{D×k} spanning dominant contrastive directions."""
    x = np.asarray(rows, dtype=np.float64)
    n, d = x.shape
    k = min(rank, n, d)
    if k == 0:
        return np.zeros((d, 0), dtype=np.float64)

    rng = np.random.default_rng(seed)
    p = min(oversampling, max(0, n - k))
    sketch_dim = k + p
    omega = rng.standard_normal((d, sketch_dim))
    y = x @ omega
    q, _ = np.linalg.qr(y, mode="reduced")
    b = q.T @ x
    u_b, s_b, vt_b = np.linalg.svd(b, full_matrices=False)
    v = vt_b[:k].T
    return v[:, :k]


def project_activation(basis: np.ndarray, activation: np.ndarray) -> np.ndarray:
    """z = P x with P = V^T — latent activation."""
    return basis.T @ np.asarray(activation, dtype=np.float64).reshape(-1)


def project_batch(basis: np.ndarray, activations: np.ndarray) -> np.ndarray:
    return activations @ basis
