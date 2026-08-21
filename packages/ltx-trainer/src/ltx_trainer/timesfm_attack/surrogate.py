"""Zero-shot TimesFM surrogate (AR predictor per channel)."""

from __future__ import annotations

import numpy as np


def fit_ar2_coefficients(series: np.ndarray) -> tuple[float, float, float]:
    """Return (b0, b1, b2) for y_k = b0 + b1 y_{k-1} + b2 y_{k-2}."""
    y = np.asarray(series, dtype=np.float64).reshape(-1)
    if len(y) < 4:
        return 0.0, 1.0, 0.0
    y1 = y[1:-1]
    y0 = y[:-2]
    y_tgt = y[2:]
    design = np.column_stack([np.ones(len(y_tgt)), y1, y0])
    coef, _, _, _ = np.linalg.lstsq(design, y_tgt, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def timesfm_predict(history: np.ndarray, coefs: np.ndarray | None = None) -> np.ndarray:
    """Predict next measurement from context window (L, m).

    Univariate AR(2) per channel — stand-in for Google TimesFM zero-shot forecast.
    If ``coefs`` is provided with shape (m, 3), use frozen clean-window coefficients.
    """
    if history.ndim != 2:
        raise ValueError("history must be (L, m)")
    l_len, m = history.shape
    pred = np.zeros(m, dtype=np.float64)
    for i in range(m):
        y = history[:, i]
        if coefs is not None:
            b0, b1, b2 = coefs[i]
            pred[i] = b0 + b1 * y[-1] + b2 * y[-2] if l_len >= 2 else y[-1]
        elif l_len >= 4:
            b0, b1, b2 = fit_ar2_coefficients(y)
            pred[i] = b0 + b1 * y[-1] + b2 * y[-2]
        elif l_len >= 2:
            pred[i] = 2.0 * y[-1] - y[-2]
        else:
            pred[i] = y[-1]
    return pred
