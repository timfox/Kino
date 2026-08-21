"""Toy reconstruction and joint training objectives (arXiv:2605.22262)."""

from __future__ import annotations

import numpy as np


def si_snr_db(estimate: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-invariant SNR in dB (toy, single-channel)."""
    s = np.asarray(estimate, dtype=np.float64).ravel()
    t = np.asarray(target, dtype=np.float64).ravel()
    if s.size != t.size or s.size == 0:
        return float("nan")
    dot = float(np.dot(s, t))
    t_norm2 = float(np.dot(t, t)) + eps
    s_target = (dot / t_norm2) * t
    e_noise = s - s_target
    num = float(np.dot(s_target, s_target)) + eps
    den = float(np.dot(e_noise, e_noise)) + eps
    return float(10.0 * np.log10(num / den))


def joint_loss(
    l_asc: float,
    l_den: float,
    lambda_asc: float = 1.0,
    lambda_den: float = 1.0,
) -> float:
    """L_tot = λ_ASC L_ASC + λ_den L_den (Eq. 4)."""
    return float(lambda_asc * l_asc + lambda_den * l_den)


def asc_cross_entropy_loss(prob_correct: float, eps: float = 1e-12) -> float:
    """Toy scalar CE for correct class probability p_c (Eq. 1)."""
    p = min(max(float(prob_correct), eps), 1.0 - eps)
    return float(-np.log(p))
