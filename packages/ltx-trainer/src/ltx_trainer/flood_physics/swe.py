"""Shallow water equation residuals — Sec. III-B (steady-state stub)."""

from __future__ import annotations

import math
from typing import Any


def central_diff(field: list[list[float]], dx: float = 1.0, dy: float = 1.0) -> tuple[list[list[float]], list[list[float]]]:
    """Second-order central differences on 2-D grid."""
    h = len(field)
    w = len(field[0]) if h else 0
    fx = [[0.0] * w for _ in range(h)]
    fy = [[0.0] * w for _ in range(h)]
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            fx[i][j] = (field[i][j + 1] - field[i][j - 1]) / (2.0 * dx)
            fy[i][j] = (field[i + 1][j] - field[i - 1][j]) / (2.0 * dy)
    return fx, fy


def manning_friction(u: float, v: float, h: float, n: float, g: float = 9.81) -> tuple[float, float]:
    """Eq. 3 bottom shear stress / ρ (stub)."""
    speed = math.hypot(u, v)
    if h <= 1e-6:
        return 0.0, 0.0
    coeff = g * n * n * speed / (h ** (1.0 / 3.0))
    return coeff * u, coeff * v


def swe_residuals(
    h: list[list[float]],
    u: list[list[float]],
    v: list[list[float]],
    zb: list[list[float]],
    *,
    n: float = 0.035,
    g: float = 9.81,
    steady_state: bool = True,
) -> dict[str, float]:
    """Compute mean squared residuals Rh, Ru, Rv — Eq. 4 (steady: ∂t=0)."""
    eta = [[zb[i][j] + h[i][j] for j in range(len(h[0]))] for i in range(len(h))]
    div_hu_x, div_hu_y = central_diff([[h[i][j] * u[i][j] for j in range(len(h[0]))] for i in range(len(h))])
    eta_x, eta_y = central_diff(eta)

    rh_sum = ru_sum = rv_sum = 0.0
    count = 0
    rows, cols = len(h), len(h[0])
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if h[i][j] <= 1e-4:
                continue
            rh = div_hu_x[i][j] + div_hu_y[i][j]
            if not steady_state:
                rh += 0.0  # ∂t h placeholder
            tbx, tby = manning_friction(u[i][j], v[i][j], h[i][j], n, g)
            ru = g * h[i][j] * eta_x[i][j] + tbx
            rv = g * h[i][j] * eta_y[i][j] + tby
            rh_sum += rh * rh
            ru_sum += ru * ru
            rv_sum += rv * rv
            count += 1
    if count == 0:
        return {"Rh_mse": 0.0, "Ru_mse": 0.0, "Rv_mse": 0.0, "mass_imbalance_pct": 0.0}
    rh_mse = rh_sum / count
    ru_mse = ru_sum / count
    rv_mse = rv_sum / count
    imbalance = 100.0 * math.sqrt(rh_mse) / max(sum(h[i][j] for i in range(rows) for j in range(cols)) / (rows * cols), 1e-6)
    return {
        "Rh_mse": rh_mse,
        "Ru_mse": ru_mse,
        "Rv_mse": rv_mse,
        "mass_imbalance_pct": min(imbalance, 100.0),
    }


def swe_card() -> dict[str, Any]:
    return {
        "equations": "depth-averaged SWE (Eq. 1–2)",
        "steady_state": "∂t(·)=0 for single-date SAR/optical",
        "friction": "Manning n from land-cover (Eq. 3)",
        "residuals": "Rh mass, Ru/Rv momentum (Eq. 4)",
        "wet_mask": "Ωw = {(x,y): h > ε}",
    }
