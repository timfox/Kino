"""Plug-in and toy classical EDL losses (arXiv:2605.22746)."""

from __future__ import annotations

import numpy as np


def plug_in_cross_entropy(p_hat: np.ndarray, y: int, eps: float = 1e-12) -> float:
    """ℓ_CE^plug(α, y) = −log Π(α)_y (Eq. 11)."""
    p = np.asarray(p_hat, dtype=np.float64)
    return float(-np.log(max(p[y], eps)))


def plug_in_mse(p_hat: np.ndarray, y: int, num_classes: int) -> float:
    """ℓ_MSE^plug(α, y) = ‖Π(α) − y‖²_2 (Eq. 11)."""
    p = np.asarray(p_hat, dtype=np.float64)
    one_hot = np.zeros(num_classes, dtype=np.float64)
    one_hot[y] = 1.0
    return float(np.sum((p - one_hot) ** 2))


def edl_mse_variance_term(p_hat: np.ndarray, alpha0: float) -> float:
    """Dirichlet variance term in classical EDL-MSE (Eq. 6), toy sum over classes."""
    p = np.asarray(p_hat, dtype=np.float64)
    return float(np.sum(p * (1.0 - p)) / (alpha0 + 1.0))


def classical_edl_mse_toy(
    p_hat: np.ndarray,
    y: int,
    alpha0: float,
    num_classes: int,
) -> float:
    """LEDL_MSE ≈ plug-in MSE + variance correction (Eq. 6)."""
    return plug_in_mse(p_hat, y, num_classes) + edl_mse_variance_term(p_hat, alpha0)


def approximation_remainder_bound(alpha0: float) -> float:
    """O((α0+1)^−1) scale for smooth-loss remainder (Theorem 2)."""
    return 1.0 / (alpha0 + 1.0)
