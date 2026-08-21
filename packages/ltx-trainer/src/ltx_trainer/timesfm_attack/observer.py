"""Luenberger observer primary detector (Eq. 6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.timesfm_attack.chi2 import chi2_statistic, chi2_threshold
from ltx_trainer.timesfm_attack.config import MassSpringConfig
from ltx_trainer.timesfm_attack.plant import mass_spring_matrices


@dataclass
class ObserverRun:
    y_hat: np.ndarray
    z: np.ndarray
    g: np.ndarray
    tau: float
    sigma_p: np.ndarray
    alarms: np.ndarray


def luenberger_gain(a: np.ndarray, c: np.ndarray, poles: tuple[complex, complex]) -> np.ndarray:
    """Place observer poles for (A - KC) via random search (2 × m gain)."""
    target = np.array(poles, dtype=np.complex128)
    rng = np.random.RandomState(0)
    best: np.ndarray | None = None
    best_err = 1e9
    for scale in (0.5, 0.25, 0.1, 0.05):
        for _ in range(400):
            k_try = rng.randn(2, c.shape[0]) * scale
            cl = a - k_try @ c
            if np.max(np.abs(np.linalg.eigvals(cl))) >= 1.0:
                continue
            eigs = np.linalg.eigvals(cl)
            err = sum(min(abs(e - t) for e in eigs) for t in target)
            if err < best_err:
                best_err = err
                best = k_try.copy()
        if best is not None:
            break
    if best is None:
        best = 0.1 * np.ones((2, c.shape[0]), dtype=np.float64)
    return best


def run_primary_detector(
    y: np.ndarray,
    cfg: MassSpringConfig,
    *,
    y_tilde: np.ndarray | None = None,
) -> ObserverRun:
    """Run observer + χ² test on measurements y (or attacked y_tilde)."""
    a, c = mass_spring_matrices(cfg)
    k = luenberger_gain(a, c, cfg.observer_poles)
    y_in = y if y_tilde is None else y_tilde
    n = y_in.shape[0]
    m = c.shape[0]
    x_hat = np.zeros(2, dtype=np.float64)
    x_hat[:] = np.array([1.0, 0.0])
    z_hist = np.zeros((n, m))
    g_hist = np.zeros(n)
    y_hat = np.zeros((n, m))

    for t in range(n):
        y_meas = y_in[t]
        z = y_meas - c @ x_hat
        z_hist[t] = z
        x_hat = a @ x_hat + k @ z
        y_hat[t] = c @ x_hat

    # Nominal residual covariance from pre-attack clean segment
    clean_end = cfg.replay_ka
    clean = z_hist[: max(clean_end, m + 1)]
    sigma_p = np.cov(clean.T, bias=False)
    if sigma_p.ndim == 0:
        sigma_p = np.array([[float(sigma_p)]])
    sigma_p = sigma_p + 1e-8 * np.eye(m)

    tau = chi2_threshold(m, cfg.alpha_p)
    for t in range(n):
        g_hist[t] = chi2_statistic(z_hist[t], sigma_p)

    alarms = g_hist > tau
    return ObserverRun(
        y_hat=y_hat,
        z=z_hist,
        g=g_hist,
        tau=tau,
        sigma_p=sigma_p,
        alarms=alarms,
    )
