"""Toy PSZ baseline and neighbor-consistency losses (arXiv:2605.21891)."""

from __future__ import annotations

import numpy as np


def clip_coordinates(x: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    lo, hi = bounds
    return np.clip(np.asarray(x, dtype=np.float64), lo, hi)


def same_region_mask(
    x: np.ndarray,
    x_prime: np.ndarray,
    x1: np.ndarray,
    dov: float,
    coord_dim: int = 2,
) -> bool:
    """Return True when both configs share overlap regime (Eq. 10–11)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x_prime = np.asarray(x_prime, dtype=np.float64).ravel()
    x1 = np.asarray(x1, dtype=np.float64).ravel()

    def regime(stacked: np.ndarray) -> bool:
        x2_coord = stacked[coord_dim : 2 * coord_dim]
        return float(np.linalg.norm(x1 - x2_coord)) > dov

    return regime(x) == regime(x_prime)


def neighbor_consistency_loss(
    g: np.ndarray,
    g_prime: np.ndarray,
    mask: bool = True,
) -> float:
    """Masked MSE between filter vectors (Eq. 12, simplified)."""
    if not mask:
        return 0.0
    g = np.asarray(g, dtype=np.float64)
    g_prime = np.asarray(g_prime, dtype=np.float64)
    if g.shape != g_prime.shape:
        raise ValueError("filter vectors must match shape")
    d = max(g.size, 1)
    return float(np.mean((g - g_prime) ** 2) / d)


def baseline_psz_loss_toy(
    bright_error: float,
    dark_energy: float,
    gain_penalty: float = 0.0,
    compact_penalty: float = 0.0,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.5,
) -> float:
    """Weighted-sum surrogate of Eq. 6 (toy scalars)."""
    l_bz = bright_error * 1e3
    l_dz = dark_energy * 1e3
    l_gain = gain_penalty
    l_compact = compact_penalty * 5.0
    return alpha * l_bz + (1.0 - alpha) * l_dz + beta * l_gain + gamma * l_compact


def total_training_loss(
    l_psz: float,
    l_nc: float,
    lam: float,
    nc_scale: float = 1e3,
) -> float:
    """L = L_psz + λ L_nc with optional numerical scale on L_nc (Eq. 13)."""
    return float(l_psz + lam * l_nc * nc_scale)
