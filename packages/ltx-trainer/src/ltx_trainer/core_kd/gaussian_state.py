"""Gaussian-inspired states and distances for CoRe-KD."""

from __future__ import annotations

import numpy as np


def state_from_logvar(
    mu: np.ndarray,
    logvar: np.ndarray,
    *,
    logvar_min: float = -6.0,
    logvar_max: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mu, sigma, precision) with clipped log-variance."""
    lv = np.clip(np.asarray(logvar, dtype=np.float64), logvar_min, logvar_max)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.exp(0.5 * lv)
    precision = np.exp(-lv)
    return mu, sigma, precision


def state_head_forward(
    hidden: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    logvar_min: float = -6.0,
    logvar_max: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lightweight MLP stub: hidden @ W -> [mu || logvar]."""
    h = np.asarray(hidden, dtype=np.float64).ravel()
    w = np.asarray(weight, dtype=np.float64)
    out = h @ w
    if bias is not None:
        out = out + np.asarray(bias, dtype=np.float64)
    d = out.size // 2
    return state_from_logvar(out[:d], out[d:], logvar_min=logvar_min, logvar_max=logvar_max)


def dg_distance(
    mu_a: np.ndarray,
    sigma_a: np.ndarray,
    mu_b: np.ndarray,
    sigma_b: np.ndarray,
) -> float:
    """Squared 2-Wasserstein proxy DG (Eq. 7)."""
    mu_a = np.asarray(mu_a, dtype=np.float64).ravel()
    mu_b = np.asarray(mu_b, dtype=np.float64).ravel()
    sigma_a = np.asarray(sigma_a, dtype=np.float64).ravel()
    sigma_b = np.asarray(sigma_b, dtype=np.float64).ravel()
    d = max(1, mu_a.size)
    return float((np.sum((mu_a - mu_b) ** 2) + np.sum((sigma_a - sigma_b) ** 2)) / d)


def kl_distill_loss(
    teacher_logits: np.ndarray,
    student_logits: np.ndarray,
    *,
    temperature: float = 2.0,
) -> float:
    """τ² D_KL(π_τ(teacher) || π_τ(student)) (Eq. 6)."""
    t = float(temperature)
    ot = np.asarray(teacher_logits, dtype=np.float64).ravel()
    os_ = np.asarray(student_logits, dtype=np.float64).ravel()
    pt = np.exp(ot / t - np.max(ot / t))
    pt /= pt.sum()
    ps = np.exp(os_ / t - np.max(os_ / t))
    ps /= ps.sum()
    eps = 1e-9
    return float(t * t * np.sum(pt * (np.log(pt + eps) - np.log(ps + eps))))
