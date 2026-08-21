"""Mixing-balance regularizer on empirical cluster mass."""

from __future__ import annotations

import numpy as np


def empirical_mass(gamma: np.ndarray) -> np.ndarray:
    """π_k = (1/N) Σ_i γ_ik."""
    return gamma.mean(axis=0)


def mixing_balance_value(pi: np.ndarray, lam: float) -> float:
    """R(π) = −λ/2 ‖π − u‖² with u = 1/K."""
    k = len(pi)
    u = 1.0 / k
    diff = pi - u
    return float(-0.5 * lam * np.dot(diff, diff))


def mixing_balance_grad(pi: np.ndarray, lam: float) -> np.ndarray:
    """∇_π R(π) = −λ(π − u)."""
    k = len(pi)
    u = 1.0 / k
    return -lam * (pi - u)
