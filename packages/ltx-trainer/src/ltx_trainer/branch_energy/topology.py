"""Three-phase topology models — Eqs. (11)–(12)."""

from __future__ import annotations

from typing import Any

import numpy as np


def delta_parallel_current(
    v_xy: np.ndarray,
    *,
    g: float,
    gamma: float,
    c: float,
    dt: float,
) -> np.ndarray:
    """ixy = G vxy + Γ v̈xy + C v'xy — Eq. (11)."""
    v_dot = np.gradient(v_xy, dt)
    v_int = np.cumsum(v_xy, dtype=np.float64) * dt
    return g * v_xy + gamma * v_int + c * v_dot


def wye_series_voltage(
    i_x: np.ndarray,
    *,
    r: float,
    l: float,
    s: float,
    dt: float,
) -> np.ndarray:
    """vxn = Rx ix + Lx i'x + Sx îx — Eq. (12), Sx = 1/Cx."""
    i_dot = np.gradient(i_x, dt)
    i_int = np.cumsum(i_x) * dt
    return r * i_x + l * i_dot + s * i_int


def virtual_neutral_voltage(v_ab: np.ndarray, v_bc: np.ndarray, v_ca: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """vxn' = ⅓(vxy − vzx) for cyclic (a,b,c)."""
    v_an = (v_ab - v_ca) / 3.0
    v_bn = (v_bc - v_ab) / 3.0
    v_cn = (v_ca - v_bc) / 3.0
    return v_an, v_bn, v_cn


def topology_card() -> dict[str, Any]:
    return {
        "delta_parallel": "line-to-line G, Γ=1/L, C per branch",
        "wye_series": "phase-to-neutral R, L, S=1/C",
        "virtual_neutral": "required for three-wire wye fit",
    }
