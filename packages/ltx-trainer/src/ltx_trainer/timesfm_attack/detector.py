"""Algorithm 1 — TimesFM-based secondary detector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.timesfm_attack.chi2 import chi2_statistic, chi2_threshold
from ltx_trainer.timesfm_attack.config import MassSpringConfig
from ltx_trainer.timesfm_attack.surrogate import fit_ar2_coefficients, timesfm_predict


@dataclass
class TimesFMDetectorRun:
    g: np.ndarray
    tau: float
    sigma: np.ndarray
    alarms: np.ndarray
    y_hat: np.ndarray


def run_timesfm_detector(
    y_tilde: np.ndarray,
    cfg: MassSpringConfig,
) -> TimesFMDetectorRun:
    """Algorithm 1: warmup → clean covariance → online χ² on forecast residuals."""
    n, m = y_tilde.shape
    l_ctx = cfg.context_length
    tw = cfg.warmup_length
    tc = cfg.clean_length
    theta = cfg.buffer_theta

    if tw + tc + l_ctx + 2 > n:
        raise ValueError("insufficient horizon for TimesFM detector phases")

    buffer = list(y_tilde[:tw].copy())
    residuals_clean: list[np.ndarray] = []

    for k in range(tw, tw + tc):
        hist = np.asarray(buffer[-l_ctx:], dtype=np.float64)
        if len(hist) < l_ctx:
            hist = np.pad(hist, ((l_ctx - len(hist), 0), (0, 0)), mode="edge")
        y_hat = timesfm_predict(hist)
        r = y_tilde[k] - y_hat
        residuals_clean.append(r)
        buffer.append(y_tilde[k].copy())

    r_mat = np.stack(residuals_clean, axis=0)
    sigma = np.cov(r_mat.T, bias=False)
    if sigma.ndim == 0:
        sigma = np.array([[float(sigma)]])
    sigma = sigma + 1e-6 * np.eye(m)
    tau = chi2_threshold(m, cfg.alpha_s)

    # Freeze AR(2) coefficients from pre-attack buffer (zero-shot surrogate).
    warm_hist = np.asarray(buffer[-l_ctx:], dtype=np.float64)
    coefs = np.zeros((m, 3), dtype=np.float64)
    for i in range(m):
        coefs[i] = fit_ar2_coefficients(warm_hist[:, i])

    # Open-loop counterfactual context: extrapolate nominal dynamics instead of
    # ingesting replayed measurements into the forecast path (TimesFM has no
    # exploitable innovation structure for the attacker to match).
    open_buffer = list(buffer)

    g = np.zeros(n)
    y_hat_hist = np.zeros((n, m))
    alarms = np.zeros(n, dtype=bool)

    for k in range(tw + tc, n):
        hist = np.asarray(open_buffer[-l_ctx:], dtype=np.float64)
        y_hat = timesfm_predict(hist, coefs=coefs)
        y_hat_hist[k] = y_hat
        r = y_tilde[k] - y_hat
        g[k] = chi2_statistic(r, sigma)
        if g[k] > tau:
            alarms[k] = True
        else:
            blended = theta * y_tilde[k] + (1.0 - theta) * y_hat
            buffer.append(blended.copy())
            open_buffer.append(y_hat.copy())

    return TimesFMDetectorRun(g=g, tau=tau, sigma=sigma, alarms=alarms, y_hat=y_hat_hist)
