"""Finite-blocklength achievability and converse rate formulas (Sec. III–IV, V)."""

from __future__ import annotations

import math

from ltx_trainer.npc.geometry import log_kl_covering_bound, log_message_count_leading


def log_m_star_converse_upper(
    n: int,
    d: int,
    lambda_star: float,
    *,
    a: float = 1.0,
    remainder: float = 0.0,
) -> float:
    """Theorem 2: ``log M*(n,ϵ) ≤ d log(√n/a) + log λ* + O_{W,ϵ,a}(1)``."""
    if n < 1 or d < 1 or a <= 0:
        raise ValueError("invalid n, d, or a")
    rho = (a * a) / n
    return log_kl_covering_bound(rho, d, lambda_star) + remainder


def log_m_star_achievability_lower(
    n: int,
    d: int,
    c: float,
    lambda_star: float,
) -> float:
    """Proposition 1 leading term: ``d/2 log n + d log c + log λ* - log d!``."""
    if n < 1 or d < 1 or c <= 0:
        raise ValueError("invalid n, d, or c")
    N = max(1, int(c * math.sqrt(n)))
    return log_message_count_leading(N, d, lambda_star)


def normalized_rate(log_m: float, n: int) -> float:
    """``log M / log n`` for rate–blocklength plots (Sec. VI)."""
    if n <= 1:
        raise ValueError("n must be > 1")
    return log_m / math.log(n)


def epsilon_capacity(d: int) -> float:
    """Strong converse ``C_ϵ = d/2`` (Corollary 1)."""
    return 0.5 * d


def gaussian_approx_log_m_bsc(n: int, c_delta_eps: float) -> float:
    """Eq. (43): ``½ log n + log c_{δ,ϵ}``."""
    if n < 1:
        raise ValueError("n >= 1")
    return 0.5 * math.log(n) + math.log(max(c_delta_eps, 1e-300))


def gaussian_approx_log_m_3x4(n: int, c_eps: float, lambda_star: float, d: int = 2) -> float:
    """Eq. (45): ``2 log(c_ϵ √n) + log λ* - log 2`` for ``d=2``."""
    if n < 1 or c_eps <= 0:
        raise ValueError("invalid n or c_eps")
    return (
        d * (math.log(c_eps) + 0.5 * math.log(n))
        + math.log(lambda_star)
        - math.lgamma(d + 1)
    )


def meta_converse_eta(epsilon: float) -> float:
    """Paper choice ``η = (1+ϵ)/2`` for Lemma 6."""
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon in (0, 1)")
    return 0.5 * (1.0 + epsilon)


def meta_converse_penalty(epsilon: float, eta: float | None = None) -> float:
    """``-log(1 - ϵ/η)`` from Lemma 6."""
    eta = eta if eta is not None else meta_converse_eta(epsilon)
    if not epsilon < eta < 1.0:
        raise ValueError("need ϵ < η < 1")
    return -math.log(1.0 - epsilon / eta)
