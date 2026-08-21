"""FAST-ME semantic-aware blended cost and adaptive stopping (Sec. IV)."""

from __future__ import annotations

import math

from ltx_trainer.fast_me.config import FastMEConfig


def blended_cost(sad: float, attention: float, *, alpha: float) -> float:
    r"""Ȳ_k = α·Y_k + (1-α)·(1 - A_k) (Eq. 28–29)."""
    a = max(0.0, min(1.0, attention))
    return alpha * sad + (1.0 - alpha) * (1.0 - a)


def adaptive_delta(delta0: float, attention: float) -> float:
    r"""δ_k = δ_0 (1 - A_k)."""
    a = max(0.0, min(1.0, attention))
    return delta0 * (1.0 - a)


def stopping_boundary(attention: float, *, delta0: float, theta: float) -> float:
    r"""T_k = -log(δ_k) / θ with δ_k = δ_0(1 - A_k) (Eq. 32)."""
    dk = adaptive_delta(delta0, attention)
    dk = max(dk, 1e-12)
    return -math.log(dk) / theta


def should_stop_fast_me(
    sad: float,
    attention: float,
    *,
    best_sad: float,
    cfg: FastMEConfig | None = None,
) -> bool:
    """Accept match when Ȳ_k <= T_k and Y_k improves best SAD (Alg. 2)."""
    cfg = cfg or FastMEConfig()
    y_tilde = blended_cost(sad, attention, alpha=cfg.alpha)
    tk = stopping_boundary(attention, delta0=cfg.delta0, theta=cfg.theta)
    return y_tilde <= tk and sad < best_sad
