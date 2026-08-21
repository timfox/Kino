"""Gradient-Guided Adaptive Sampling (GGAS) — Sec. IV-D, Eq. (7)–(10)."""

from __future__ import annotations

import math


def retrieval_token_difficulty(
    grad_retq_norm: float,
    grad_rett_norm: float,
) -> float:
    r"""d_k^(t) = ‖∇_[RETq] L‖_2 + ‖∇_[RETt] L‖_2 (Eq. 7)."""
    return grad_retq_norm + grad_rett_norm


def ema_difficulty(prev_g: float, d_k: float, *, alpha: float = 0.9) -> float:
    r"""G_k^(t) = α G_k^(t-1) + (1-α) d_k^(t) (Eq. 8)."""
    return alpha * prev_g + (1.0 - alpha) * d_k


def sampling_score(
    g_k: float,
    n_k: float,
    *,
    eta: float = 1.0,
    gamma: float = 0.5,
) -> float:
    r"""S_k^(t) = exp(G_k^(t) / η + γ log N_k) (Eq. 9)."""
    return math.exp(g_k / eta + gamma * math.log(max(n_k, 1.0)))


def sampling_probabilities(
    scores: list[float],
    *,
    eps: float = 0.02,
) -> list[float]:
    r"""P̃_k = max(S_k / Σ S, ε); P_k = P̃_k / Σ P̃ (Eq. 10)."""
    if not scores:
        return []
    s_sum = sum(scores) or 1.0
    raw = [max(s / s_sum, eps) for s in scores]
    z = sum(raw) or 1.0
    return [p / z for p in raw]


def argmax_task_index(probs: list[float]) -> int:
    """Batch-level pick k* = argmax_k P_k (toy)."""
    if not probs:
        return 0
    return max(range(len(probs)), key=lambda i: probs[i])
