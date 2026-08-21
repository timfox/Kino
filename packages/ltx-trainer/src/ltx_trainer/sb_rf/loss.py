"""Composite training loss (Eq. 11)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.sb_rf.flow import velocity_matching_loss


def amplitude_transform(c: np.ndarray, alpha: float = 0.5, beta: float = 0.33) -> np.ndarray:
    """˜c = β |c|^α e^{j∠c} (Sec. 4.2)."""
    mag = np.abs(c)
    phase = np.angle(c)
    return beta * (mag**alpha) * np.exp(1j * phase)


def mel_proxy_loss(x_hat: np.ndarray, x: np.ndarray) -> float:
    """Lightweight magnitude L1 proxy for multi-res mel loss."""
    return float(np.mean(np.abs(np.abs(x_hat) - np.abs(x))))


def pesq_proxy_loss(x_hat: np.ndarray, x: np.ndarray) -> float:
    """Lightweight spectral L2 proxy for PESQ loss."""
    return float(np.mean(np.abs(x_hat - x) ** 2))


def composite_loss(
    v_pred: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    xt: np.ndarray,
    t: float,
    *,
    epsilon: float = 0.03,
    lambda_mel: float = 33.0,
    lambda_pesq: float = 3.0,
) -> dict[str, float]:
    from ltx_trainer.sb_rf.flow import estimate_clean

    lv = velocity_matching_loss(v_pred, x, y)
    x_hat = estimate_clean(xt, t, v_pred, epsilon)
    lmel = mel_proxy_loss(x_hat, x)
    lpesq = pesq_proxy_loss(x_hat, x)
    total = lv + lambda_mel * lmel + lambda_pesq * lpesq
    return {"total": total, "velocity": lv, "mel": lmel, "pesq": lpesq}
