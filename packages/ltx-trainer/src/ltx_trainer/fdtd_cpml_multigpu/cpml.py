"""CFS/Roden–Gedney CPML coefficients (Section 3.2)."""

from __future__ import annotations

import math
from typing import Any

from ltx_trainer.fdtd_cpml_multigpu.constants import (
    ALPHA_FRAC,
    KAPPA_MAX,
    L_CPML,
    R_TARGET,
)


def cpml_profiles(
    *,
    rho: float,
    h: float,
    l_cpml: int = L_CPML,
    r_target: float = R_TARGET,
) -> dict[str, float]:
    """Cubic grading q=3 profiles at normalized depth rho ∈ [0,1]."""
    rho_q = rho**3
    sigma_max = -4.0 * math.log(r_target) / (2.0 * l_cpml * h)
    alpha_max = ALPHA_FRAC * sigma_max
    sigma = sigma_max * rho_q
    kappa = 1.0 + (KAPPA_MAX - 1.0) * rho_q
    alpha = alpha_max * (1.0 - rho)
    return {
        "sigma": sigma,
        "kappa": kappa,
        "alpha": alpha,
        "sigma_max": sigma_max,
        "alpha_max": alpha_max,
    }


def recursive_coefficients(
    *,
    sigma: float,
    kappa: float,
    alpha: float,
    dt: float,
) -> dict[str, float]:
    bx = math.exp(-((sigma / kappa) + alpha) * dt)
    denom = kappa * (sigma + kappa * alpha)
    ax = sigma * (bx - 1.0) / denom if abs(denom) > 1e-30 else 0.0
    return {"bx": bx, "ax": ax}


def cpml_card(h: float = 1.0, dt: float = 0.001) -> dict[str, Any]:
    edge = cpml_profiles(rho=1.0, h=h)
    mid = cpml_profiles(rho=0.5, h=h)
    rec = recursive_coefficients(sigma=edge["sigma"], kappa=edge["kappa"], alpha=edge["alpha"], dt=dt)
    return {
        "l_cpml_cells": L_CPML,
        "r_target": R_TARGET,
        "grading": "cubic q=3",
        "edge_profiles": edge,
        "mid_profiles": mid,
        "recursive_at_edge": rec,
        "fields": ["p", "vx", "vy", "vz", "psi_p_*", "psi_v_*"],
        "application": "global outer boundary only; internal GPU halos use exchange only",
    }
