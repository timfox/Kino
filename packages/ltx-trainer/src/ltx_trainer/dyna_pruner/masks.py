"""Importance maps, data masks, and STE binarization (Sec. III-C)."""

from __future__ import annotations

from typing import Any

import numpy as np


def importance_from_temporal_variance(x: np.ndarray) -> np.ndarray:
    """
    Lightweight mask-generator proxy: per-pixel temporal variance across T (and mean over C).

    ``x`` shape ``(T, C, H, W)`` → ``S`` in ``[0, 1]^{H×W}``.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError(f"expected (T,C,H,W), got {arr.shape}")
    var = np.var(arr, axis=(0, 1))
    vmax = float(np.max(var)) or 1.0
    return np.clip(var / vmax, 0.0, 1.0)


def soft_data_mask(s: np.ndarray) -> np.ndarray:
    """Training-time soft mask (X' = X ⊙ S)."""
    return np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)


def hard_mask_ste(s: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    """
    STE binarization for inference masks.

    Forward pass returns hard {0,1}; training backward would route through soft ``S``.
    """
    soft = soft_data_mask(s)
    return (soft >= threshold).astype(np.float64)


def apply_data_mask(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply spatial mask to ``(T,C,H,W)`` via broadcast on H,W."""
    arr = np.asarray(x, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError("mask must be H×W")
    return arr * m[np.newaxis, np.newaxis, :, :]


def sparsity_fraction(mask: np.ndarray) -> float:
    """Fraction of elements pruned (1 - mean)."""
    m = np.asarray(mask, dtype=np.float64)
    return float(1.0 - np.mean(m))


def l1_sparsity_penalty(s: np.ndarray) -> float:
    """L1 proxy for score map / unit importances (Eq. 4)."""
    return float(np.sum(np.abs(np.asarray(s, dtype=np.float64))))


def mask_summary(s: np.ndarray, *, threshold: float = 0.5) -> dict[str, Any]:
    soft = soft_data_mask(s)
    hard = hard_mask_ste(soft, threshold=threshold)
    return {
        "mean_importance": round(float(np.mean(soft)), 4),
        "data_sparsity": round(sparsity_fraction(hard), 4),
        "active_pixels": int(np.sum(hard > 0.5)),
        "total_pixels": int(hard.size),
    }
