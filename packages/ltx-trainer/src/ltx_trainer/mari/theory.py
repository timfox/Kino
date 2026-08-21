"""Theorem 5.1–5.2 reference bounds (Appendix B)."""

from __future__ import annotations


def routing_risk_bound(
    r_min: float,
    *,
    misrouting_rate: float,
    loss_bound: float = 1.0,
) -> float:
    """Eq. 17: R_ent <= R_min + L * η."""
    return r_min + loss_bound * misrouting_rate


def improvement_condition(
    r_single: float,
    r_min: float,
    *,
    misrouting_rate: float,
    loss_bound: float = 1.0,
) -> dict[str, float | bool]:
    """Eq. 18: Δ_spec > L η => R_ent < R_single."""
    delta_spec = r_single - r_min
    threshold = loss_bound * misrouting_rate
    return {
        "delta_spec": delta_spec,
        "L_eta": threshold,
        "improves_over_single": delta_spec > threshold,
    }


def non_applicable_energy_bound(
    alpha: float,
    *,
    kappa_non: float,
    S: float,
    gamma_leak: float,
    epsilon: float,
) -> float:
    """Eq. 19 leading term: α (κ_non S + Γ ε)."""
    return alpha * (kappa_non * S + gamma_leak * epsilon)
