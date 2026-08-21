"""Branch-level power terms — Eqs. (1)–(7)."""

from __future__ import annotations

import numpy as np


def joule_dissipation_r(i: np.ndarray, r: float) -> np.ndarray:
    """p_R,r = R i_r² — Eq. (4)."""
    return r * i * i


def joule_dissipation_g(v: np.ndarray, g: float) -> np.ndarray:
    """p_R,r = G v_r² — Eq. (4) alternate."""
    return g * v * v


def magnetic_storage_rate(i: np.ndarray, l: float, dt: float) -> np.ndarray:
    """W'_L,ℓ = L i_ℓ i'_ℓ — Eq. (5)."""
    di = np.gradient(i, dt)
    return l * i * di


def electric_storage_rate(v: np.ndarray, c: float, dt: float) -> np.ndarray:
    """W'_C,c = C v_c v'_c — Eq. (6)."""
    dv = np.gradient(v, dt)
    return c * v * dv


def magnetic_energy(i: np.ndarray, l: float) -> np.ndarray:
    """W_L = ½ L i² — Eq. (2)."""
    return 0.5 * l * i * i


def electric_energy(v: np.ndarray, c: float) -> np.ndarray:
    """W_C = ½ C v² — Eq. (2)."""
    return 0.5 * c * v * v


def terminal_power(v: np.ndarray, i: np.ndarray) -> np.ndarray:
    """p_term = v⊤i (single branch or aggregated scalar)."""
    return v * i


def branch_balance_residual(
    p_term: np.ndarray,
    p_r: np.ndarray,
    w_l_dot: np.ndarray,
    w_c_dot: np.ndarray,
) -> float:
    """‖p_term − (p_R + W'_L + W'_C)‖ / ‖p_term‖ — Eq. (3)."""
    recon = p_r + w_l_dot + w_c_dot
    num = float(np.linalg.norm(p_term - recon))
    den = float(np.linalg.norm(p_term)) + 1e-30
    return num / den


def branch_card() -> dict[str, str]:
    return {
        "joule": "p_R = Σ R_r i_r²",
        "magnetic_rate": "W'_L = Σ L_ℓ i_ℓ i'_ℓ",
        "electric_rate": "W'_C = Σ C_c v_c v'_c",
        "balance": "p_term = p_R + W'_L + W'_C (Tellegen)",
    }
