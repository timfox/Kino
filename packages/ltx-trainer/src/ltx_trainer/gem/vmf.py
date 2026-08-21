"""von Mises–Fisher mixture helpers on the unit hypersphere."""

from __future__ import annotations

import math

import numpy as np


def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def log_vmf_unnormalized(x: np.ndarray, mu: np.ndarray, kappa: float) -> np.ndarray:
    """Directional log-density up to log C_d(kappa) (constant per cluster at fixed kappa)."""
    return float(kappa) * (x @ mu)


def kappa_from_r_bar(r_bar: float, dim: int, eps: float = 1e-8) -> float:
    """High-dimensional concentration MLE (Eq. 16 in GEM paper)."""
    r = float(np.clip(r_bar, 0.0, 1.0 - eps))
    num = r * (dim - r * r)
    den = 1.0 - r * r
    return max(0.0, num / max(den, eps))


def m_step(
    x: np.ndarray,
    gamma: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form vMF M-step: mean directions and concentrations."""
    n, k = gamma.shape
    dim = x.shape[1]
    mu = np.zeros((k, dim), dtype=np.float64)
    kappa = np.zeros(k, dtype=np.float64)
    for j in range(k):
        w = gamma[:, j]
        mass = float(w.sum())
        if mass < eps:
            mu[j] = np.zeros(dim)
            mu[j, 0] = 1.0
            kappa[j] = 0.0
            continue
        r_vec = (w[:, None] * x).sum(axis=0)
        r_norm = float(np.linalg.norm(r_vec))
        mu[j] = r_vec / (r_norm + eps)
        r_bar = r_norm / mass
        kappa[j] = min(200.0, kappa_from_r_bar(r_bar, dim, eps=eps))
    return mu, kappa


def spherical_kmeans_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Pick k random data points as initial mean directions."""
    idx = rng.choice(len(x), size=min(k, len(x)), replace=False)
    return normalize_rows(x[idx].copy())
