"""Augmented-Lagrangian inspired loss terms (Eq. 12) — toy, non-training stub."""

from __future__ import annotations

import numpy as np


def si_sdr(x_hat: np.ndarray, x: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-invariant SDR in dB (time-domain)."""
    x_hat = np.asarray(x_hat, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    if x_hat.shape != x.shape:
        raise ValueError("x_hat and x must match shape")
    if x.size == 0:
        return 0.0
    s = np.dot(x_hat, x) / (np.dot(x, x) + eps)
    x_target = s * x
    e = x_hat - x_target
    num = np.dot(x_target, x_target)
    den = np.dot(e, e) + eps
    return float(10.0 * np.log10((num + eps) / den))


def pass_penalty(w: np.ndarray, a_target: np.ndarray) -> float:
    """λ_pass E_k[|w^H a - 1|^2] term (without λ)."""
    w = np.asarray(w)
    a = np.asarray(a_target)
    if w.shape != a.shape:
        raise ValueError("w and a_target must match shape (M,)")
    val = np.vdot(w, a)  # w^H a
    return float(np.abs(val - 1.0) ** 2)


def null_penalty_db(w: np.ndarray, a_interf: np.ndarray, eps: float = 1e-8) -> float:
    """λ_null E_k[10 log10(||w^H A_i||^2 + eps)] term (without λ)."""
    w = np.asarray(w)
    ai = np.asarray(a_interf)
    if w.ndim != 1:
        raise ValueError("w must be (M,)")
    if ai.ndim != 2 or ai.shape[0] != w.shape[0]:
        raise ValueError("a_interf must be (M, J-1)")
    proj = w.conj().T @ ai  # (J-1,)
    power = float(np.sum(np.abs(proj) ** 2))
    return float(10.0 * np.log10(power + float(eps)))


def schedule_lambdas(epoch: int, *, warmup_epochs: int, final: float) -> float:
    """Simple monotone schedule: 0 until warmup, then linear to `final` by 2×warmup."""
    e = int(epoch)
    if e < warmup_epochs:
        return 0.0
    span = max(1, warmup_epochs)
    t = min(1.0, (e - warmup_epochs) / float(span))
    return float(final * t)


def total_loss(
    x_hat: np.ndarray,
    x_target: np.ndarray,
    *,
    w: np.ndarray,
    a_target: np.ndarray,
    a_interf: np.ndarray,
    lambda_pass: float,
    lambda_null: float,
    null_eps: float = 1e-8,
) -> dict[str, float]:
    """Eq. (12) structure (signs kept): L = -SI-SDR + λ_pass*pass + λ_null*null_db."""
    s = si_sdr(x_hat, x_target)
    lp = pass_penalty(w, a_target)
    ln = null_penalty_db(w, a_interf, eps=null_eps)
    return {
        "si_sdr_db": s,
        "L_recon": float(-s),
        "L_pass": float(lambda_pass * lp),
        "L_null": float(lambda_null * ln),
        "L_total": float(-s + lambda_pass * lp + lambda_null * ln),
    }

