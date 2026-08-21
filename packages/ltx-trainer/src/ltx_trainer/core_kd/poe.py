"""Product-of-Experts fusion for Gaussian-inspired states (Eq. 3, C.10)."""

from __future__ import annotations

import numpy as np


def poe_fuse(
    mus: list[np.ndarray],
    precisions: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse modality states with unit prior precision."""
    if not mus:
        raise ValueError("poe_fuse requires at least one modality state")
    d = np.asarray(mus[0], dtype=np.float64).size
    precision_sum = np.ones(d, dtype=np.float64)
    weighted_mu = np.zeros(d, dtype=np.float64)
    for mu, kappa in zip(mus, precisions, strict=True):
        mu = np.asarray(mu, dtype=np.float64).ravel()
        kappa = np.asarray(kappa, dtype=np.float64).ravel()
        precision_sum += kappa
        weighted_mu += kappa * mu
    var = 1.0 / precision_sum
    mu_fused = var * weighted_mu
    sigma_fused = np.sqrt(var)
    return mu_fused, sigma_fused
