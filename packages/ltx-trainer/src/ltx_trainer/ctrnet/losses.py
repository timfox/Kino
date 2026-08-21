"""Mixture-constraint and supervised loss toys (Eqs. 8–9, 16, 28, 31)."""

from __future__ import annotations

import numpy as np


def magnitude_compressed_components(y: np.ndarray, y_hat: np.ndarray, *, alpha: float) -> np.ndarray:
    """Per-T-F terms inside G(·,·) in Eq. (9) — magnitude, cos, sin branches."""
    mag = np.abs(y) ** alpha - np.abs(y_hat) ** alpha
    ang_y = np.angle(y)
    ang_h = np.angle(y_hat)
    cos_term = np.abs(y) ** alpha * np.cos(ang_y) - np.abs(y_hat) ** alpha * np.cos(ang_h)
    sin_term = np.abs(y) ** alpha * np.sin(ang_y) - np.abs(y_hat) ** alpha * np.sin(ang_h)
    return np.abs(mag) + np.abs(cos_term) + np.abs(sin_term)


def g_loss(
    y: np.ndarray,
    y_hat: np.ndarray,
    *,
    alpha: float,
    normalize_den: np.ndarray | None = None,
) -> float:
    """Eq. (8)–(9): compressed RI magnitude loss with optional normalization."""
    num = float(np.sum(magnitude_compressed_components(y, y_hat, alpha=alpha)))
    if normalize_den is None:
        return num
    den = float(np.sum(np.abs(normalize_den) ** alpha))
    if den <= 0.0:
        return num
    return num / den


def speaker_activity_loss(
    z_hat: np.ndarray,
    activity_mask: np.ndarray,
    *,
    alpha: float,
    mixture_ref: np.ndarray,
) -> float:
    """Eq. (15): push inactive frames toward zero."""
    silent = 1.0 - activity_mask
    num = float(np.sum(np.abs(z_hat) ** alpha * silent))
    den = float(np.sum(np.abs(mixture_ref) ** alpha))
    if den <= 0.0:
        return num
    return num / den


def overlap_sampling_weight(active_counts_per_frame: np.ndarray, *, theta: float) -> float:
    """Eq. (34): training block sampling weight from speaker overlap."""
    if active_counts_per_frame.size == 0:
        return 1.0
    per_frame = np.maximum(1.0, active_counts_per_frame) - 1.0
    return 1.0 + theta * float(np.mean(per_frame))
