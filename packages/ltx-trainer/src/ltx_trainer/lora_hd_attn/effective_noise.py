"""Effective-noise mechanism — Eqs. (25)–(28)."""

from __future__ import annotations

from typing import Any


def delta_eff_independent(delta: float, Q0: float, M: float, Q: float) -> float:
    """Eq. (25): Delta_eff = Delta/2 + Q0 - 2M + Q (linear sigma, e=0)."""
    return float(delta / 2.0 + Q0 - 2.0 * M + Q)


def delta_eff_recalibrated(delta: float, Q0: float, M: float, Q: float, beta: float) -> float:
    """Eq. (26): Delta_eff(beta) with Q -> beta^2 Q, M -> beta M."""
    return float(delta / 2.0 + Q0 + beta * beta * Q - 2.0 * beta * M)


def beta_min(M: float, Q: float) -> float:
    """Eq. (26): beta_min = M / Q."""
    return float(M / max(Q, 1e-12))


def delta_eff_at_beta_min(delta: float, Q0: float, M: float, Q: float) -> float:
    """Eq. (26): Delta_eff(beta_min) = Delta/2 + Q0 - M^2/Q."""
    return float(delta / 2.0 + Q0 - (M * M) / max(Q, 1e-12))


def delta_eff_reused(delta_eff_0: float, V: float, T: int) -> float:
    """Eq. (27): reused sequences e=1."""
    denom = (1.0 + 4.0 * V / max(T * T, 1)) ** 2
    return float(delta_eff_0 / max(denom, 1e-12))


def delta_eff_active_ft(delta: float, Q0: float, M: float, Q: float) -> float:
    """Eq. (28): Delta_eff,ActFT = Delta/2 + Q0 - M^2/Q."""
    return float(delta_eff_at_beta_min(delta, Q0, M, Q))


def effective_noise_card(
    *,
    delta: float,
    Q0: float,
    M: float,
    Q: float,
    V: float,
    T: int,
    e: int,
) -> dict[str, Any]:
    de0 = delta_eff_independent(delta, Q0, M, Q)
    de1 = delta_eff_reused(de0, V, T) if e == 1 else de0
    bmin = beta_min(M, Q)
    return {
        "delta_eff_e0": de0,
        "delta_eff_e1": de1,
        "delta_eff_active_ft": delta_eff_active_ft(delta, Q0, M, Q),
        "beta_min": bmin,
        "delta_eff_beta_min": delta_eff_at_beta_min(delta, Q0, M, Q),
        "recalibration": "W -> sqrt(beta) W freezes extensive-rank scale",
    }
