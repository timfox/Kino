"""Generalized energetic duality — Corollary 2, Eqs. (8)–(10)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.branch_energy.branch import electric_storage_rate, joule_dissipation_g


def _time_primitive(x: np.ndarray, dt: float) -> np.ndarray:
    """˘x — cumulative time integral (quasi-static lumped convention)."""
    return np.cumsum(x, dtype=np.float64) * dt


def parallel_branch_energy(
    v: np.ndarray,
    i: np.ndarray,
    *,
    g: float,
    gamma: float,
    c: float,
    dt: float,
) -> dict[str, np.ndarray]:
    """Parallel G–Γ–C branch energies — Eqs. (4)–(6) with Γ = 1/L."""
    v_dot = np.gradient(v, dt)
    v_int = _time_primitive(v, dt)
    return {
        "p_R": joule_dissipation_g(v, g),
        "W_L_dot": gamma * v_int * v if abs(gamma) > 0.0 else np.zeros_like(v),
        "W_C_dot": electric_storage_rate(v, c, dt) if c > 0.0 else np.zeros_like(v),
    }


def series_from_parallel(
    v: np.ndarray,
    i: np.ndarray,
    *,
    g_p: float,
    gamma_p: float,
    c_p: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rs(t)=v²Gp/i², Ls(t)=v̘v Γp/(ii′), Ss(t)=vv′ Cp/(iî) — Cor. 2."""
    eps = 1e-9
    i_dot = np.gradient(i, dt)
    v_dot = np.gradient(v, dt)
    v_int = _time_primitive(v, dt)
    i_int = _time_primitive(i, dt)

    rs = (v * v * g_p) / (i * i + eps)
    ls = (v_int * v * gamma_p) / (i * i_dot + eps)
    ss = (v * v_dot * c_p) / (i * i_int + eps)
    return rs, ls, ss


def series_branch_energy(
    v: np.ndarray,
    i: np.ndarray,
    rs: np.ndarray,
    ls: np.ndarray,
    ss: np.ndarray,
    dt: float,
) -> dict[str, np.ndarray]:
    """Branch energies via series parameterization — Eqs. (8)–(10)."""
    i_dot = np.gradient(i, dt)
    i_int = _time_primitive(i, dt)
    return {
        "p_R": rs * i * i,
        "W_L_dot": ls * i * i_dot,
        "W_C_dot": ss * i * i_int,
    }


def duality_invariant_error(par: dict[str, np.ndarray], ser: dict[str, np.ndarray]) -> float:
    """Max |parallel − series| branch energy components (ignore near-singular samples)."""
    diffs = []
    for key in ("p_R", "W_L_dot", "W_C_dot"):
        delta = np.abs(par[key] - ser[key])
        scale = np.maximum(np.abs(par[key]), 1e-6)
        mask = scale > 1e-3
        if np.any(mask):
            diffs.append(float(np.max(delta[mask])))
        else:
            diffs.append(float(np.max(delta)))
    return max(diffs)
