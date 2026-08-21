"""Log-domain MSE + stride-based slope penalty (Eq. 1–2, arXiv:2605.20968)."""

from __future__ import annotations

import numpy as np


def to_db(y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """y_dB = 10 log10(y + eps)."""
    a = np.maximum(np.asarray(y, dtype=np.float64), 0.0)
    return 10.0 * np.log10(a + float(eps))


def slope_d_db(y_db: np.ndarray, stride_k: int) -> np.ndarray:
    """Δy_dB[n] = y_dB[n+k] − y_dB[n] — Eq. (2)."""
    y = np.asarray(y_db, dtype=np.float64).ravel()
    k = int(stride_k)
    if k <= 0:
        raise ValueError("stride_k must be positive")
    if y.size <= k:
        return np.array([], dtype=np.float64)
    return y[k:] - y[:-k]


def composite_loss(
    y_hat: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 0.2,
    stride_k: int = 50,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Lt = MSE(ŷ_dB, y_dB) + α MSE(Δŷ_dB, Δy_dB) — Eq. (1)."""
    y_hat_db = to_db(y_hat, eps=eps)
    y_db = to_db(y, eps=eps)
    mse_level = float(np.mean((y_hat_db - y_db) ** 2))

    d_hat = slope_d_db(y_hat_db, stride_k)
    d_y = slope_d_db(y_db, stride_k)
    if d_hat.size == 0:
        mse_slope = 0.0
    else:
        mse_slope = float(np.mean((d_hat - d_y) ** 2))

    return {
        "mse_db": mse_level,
        "mse_slope_db": mse_slope,
        "L_total": float(mse_level + float(alpha) * mse_slope),
    }
