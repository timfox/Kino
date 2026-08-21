"""PAC-Bayes sharpness and norm terms (Theorems 2, 5, 16–17)."""

from __future__ import annotations

import math


def gu_unperturbed(omega: int, degree: int, a: float = 4.0, b: float = 32.0) -> float:
    """Theorem 5: Tr(∇²L) ≤ 2Gu with Gu = a + b·ω·D³_f ∈ O(ω D³_f)."""
    return a + b * omega * (float(degree) ** 3)


def norm_proxy_l(
    omega: int,
    degree: int,
    context_length: int,
    scale_df: float = 16.0,
    scale_omega_df: float = 4.0,
    scale_log: float = 4.0,
) -> float:
    """Theorem 10: L(ω, Df, T) ∈ O(D³_f + log(T)² ω Df)."""
    df3 = float(degree) ** 3
    log_t = math.log2(max(context_length, 2)) ** 2
    return scale_df * df3 + scale_omega_df * omega * degree + scale_log * log_t


def semi_analytic_bound(
    m: int,
    sigma: float,
    omega: int,
    degree: int,
    context_length: int,
    sigma_sg: float = 0.01,
    delta: float = 0.05,
    p_empirical: float = 0.0,
) -> float:
    """Semi-analytic generalization bound (Theorem 2, empirical P term)."""
    gu = gu_unperturbed(omega, degree)
    l_norm = norm_proxy_l(omega, degree, context_length)
    sharp_term = sigma * sigma * (gu + p_empirical)
    norm_term = 2.0 * math.sqrt(sigma_sg * sigma_sg / (2.0 * m)) * math.sqrt(
        l_norm / (2.0 * sigma * sigma) + math.log(1.0 / delta)
    )
    return sharp_term + norm_term


def cot_bound(
    T: int,
    m: int,
    sigma: float,
    sigma_sg: float,
    gu: float,
    l_norm: float,
    p_empirical: float = 0.0,
) -> float:
    """Theorem 16: BCoT(σ) — parity with chain-of-thought, linear in T."""
    exp_inner = math.exp(-m / (8.0 * sigma_sg * sigma_sg))
    sharp = math.exp(m * sigma * sigma / (4.0 * sigma_sg * sigma_sg)) * (2.0 * gu + p_empirical)
    norm = l_norm / (2.0 * sigma * sigma)
    return 4.0 * T * exp_inner * (sharp + norm)


def onepass_bound(
    T: int,
    m: int,
    sigma: float,
    sigma_sg: float,
    gu: float,
    l_norm: float,
    p_empirical: float = 0.0,
) -> float:
    """Theorem 17/18: one-pass parity — unfavorable scaling in T."""
    exp_inner = math.exp(-m / (8.0 * sigma_sg * sigma_sg))
    sharp = math.exp(m * sigma * sigma / (4.0 * sigma_sg * sigma_sg)) * (2.0 * gu + p_empirical)
    norm = l_norm / (2.0 * sigma * sigma)
    return 4.0 * exp_inner * (sharp + norm)