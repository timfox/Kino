"""Latent Hamiltonian Hx(μ) = reconstruction + prior (Eq. 4, 15)."""

from __future__ import annotations

import numpy as np


def normalize_sphere(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        n = np.linalg.norm(z) or 1.0
        return z / n
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return z / norms


def latent_energy(
    mu: np.ndarray,
    *,
    target: np.ndarray | None = None,
    field: np.ndarray | None = None,
    lam: float = 1.0,
) -> float:
    """Scalar energy proxy: MSE reconstruction + quadratic prior − λ⟨m, μ⟩ on the sphere."""
    mu = np.asarray(mu, dtype=np.float64).ravel()
    recon = 0.0 if target is None else float(np.mean((mu - target) ** 2))
    if field is None:
        prior = 0.5 * float(np.dot(mu, mu))
        return recon + lam * prior
    field = np.asarray(field, dtype=np.float64).ravel()
    prior = 0.5 * float(np.dot(mu - field, mu - field))
    field_term = -lam * float(np.dot(field, mu))
    return recon + prior + field_term


def field_direction(mode: str, dim: int) -> np.ndarray:
    if mode == "zero":
        v = np.zeros(dim, dtype=np.float64)
        v[0] = 1.0
    elif mode == "half":
        v = np.ones(dim, dtype=np.float64)
    else:
        v = np.zeros(dim, dtype=np.float64)
        v[-1] = 1.0
    return normalize_sphere(v)
