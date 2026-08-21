"""Training losses for EntroAD (Sec. 4.4, Eq. 13–19)."""

from __future__ import annotations

import numpy as np


def binary_cross_entropy_image(y: float, y_hat: float, *, eps: float = 1e-8) -> float:
    """Eq. (13)."""
    y_hat = float(np.clip(y_hat, eps, 1.0 - eps))
    y = float(y)
    return float(-(y * np.log(y_hat + eps) + (1.0 - y) * np.log(1.0 - y_hat + eps)))


def focal_loss(pred: np.ndarray, target: np.ndarray, *, alpha: float = 0.25, gamma: float = 2.0) -> float:
    """Eq. (14) mean focal loss."""
    p = np.clip(np.asarray(pred, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    t = np.asarray(target, dtype=np.float64)
    pos = -alpha * t * ((1.0 - p) ** gamma) * np.log(p)
    neg = -(1.0 - alpha) * (1.0 - t) * (p**gamma) * np.log(1.0 - p)
    return float(np.mean(pos + neg))


def dice_loss(pred: np.ndarray, target: np.ndarray, *, eps: float = 1e-8) -> float:
    """Eq. (15)."""
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    inter = float((p * t).sum())
    return float(1.0 - (2.0 * inter + eps) / (p.sum() + t.sum() + eps))


def segmentation_loss(pred: np.ndarray, target: np.ndarray, *, lambda_d: float = 1.0) -> float:
    """Eq. (16)."""
    return focal_loss(pred, target) + lambda_d * dice_loss(pred, target)


def stage2_branch_loss(
    m_anom: np.ndarray,
    m_norm: np.ndarray,
    mask: np.ndarray,
    *,
    lambda_d: float = 1.0,
) -> float:
    """Eq. (18): focal+dice on anomaly map; dice on normal map vs 1-mask."""
    return segmentation_loss(m_anom, mask, lambda_d=lambda_d) + lambda_d * dice_loss(
        m_norm, 1.0 - np.asarray(mask)
    )
