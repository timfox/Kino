"""Replay and model-based stealthy attacks (Section II–III)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.timesfm_attack.chi2 import attack_budget_delta_tau
from ltx_trainer.timesfm_attack.config import MassSpringConfig
from ltx_trainer.timesfm_attack.plant import mass_spring_matrices


def apply_replay_attack(y: np.ndarray, cfg: MassSpringConfig) -> np.ndarray:
    """Model-free replay on all sensors (Eq. 4)."""
    out = y.copy()
    k0, k1, ka = cfg.replay_k0, cfg.replay_k1, cfg.replay_ka
    span = k1 - k0 + 1
    for k in range(ka, len(out)):
        out[k] = y[k0 + (k - ka) % span]
    return out


def stealthy_attack_lti(
    y: np.ndarray,
    cfg: MassSpringConfig,
    *,
    sigma_p: np.ndarray,
    k_gain: np.ndarray,
    tau_p: float,
) -> np.ndarray:
    """Theorem 1 closed-form additive attack a*[k] on outputs."""
    a_mat, c = mass_spring_matrices(cfg)
    n, m = y.shape
    ka, k2 = cfg.stealthy_ka, cfg.stealthy_k2
    w = np.asarray(cfg.impact_weight, dtype=np.float64).reshape(-1)
    if w.shape[0] != a_mat.shape[0]:
        w = np.array([1.0, 0.0])

    delta_tau = min(cfg.delta_tau_p, attack_budget_delta_tau(m, tau_p, cfg.alpha_p, 0.0))
    out = y.copy()
    x_hat = np.array([1.0, 0.0])
    eps = np.zeros(2)

    for k in range(n):
        z_nom = y[k] - c @ x_hat
        if ka <= k <= k2:
            c_k = k_gain.T @ np.linalg.matrix_power(a_mat, k2 - k) @ w
            denom = float(np.sqrt(max(c_k @ sigma_p @ c_k, 1e-12)))
            delta = np.sqrt(max(delta_tau, 0.0)) * (sigma_p @ c_k) / denom
            a_k = c @ eps + delta
            y_att = y[k] + a_k
            z_a = y_att - c @ x_hat
            eps = a_mat @ eps + k_gain @ (z_a - z_nom)
            out[k] = y_att
        x_hat = a_mat @ x_hat + k_gain @ (out[k] - c @ x_hat)
    return out
