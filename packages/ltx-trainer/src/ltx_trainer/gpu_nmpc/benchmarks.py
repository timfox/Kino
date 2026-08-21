"""Benchmark plant models (Sec. 4, Code Snippet 1)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.gpu_nmpc.config import DistillationParams, HeatedPlateParams


def scalar_plant_step(y: float, u: float, dt: float) -> float:
    """Code Snippet 1: dy = 4 sin(u) - y^3."""
    dy = 4.0 * np.sin(u) - y**3
    return float(y + dt * dy)


def distillation_tray1_step(x1: float, u: float, *, params: DistillationParams, dt: float) -> float:
    """Single-step proxy for tray-1 liquid mole fraction dynamics."""
    alpha = params.relative_volatility
    y2 = alpha * x1 / (1.0 + (alpha - 1.0) * x1)
    v = u * params.distillate_flow
    dx = v * (y2 - x1) / params.ac
    return float(np.clip(x1 + dt * dx, 0.0, 1.0))


def plate_average_step(t_avg: float, q: float, *, params: HeatedPlateParams, dt: float, t_set: float) -> float:
    """Proxy for average temperature tracking in Problem (30)."""
    heat = 4.1 * np.sqrt(max(q, 0.0))
    loss = params.heat_loss * (t_avg - params.boundary_temp)
    dtemp = params.thermal_conductivity * 1e-4 * (heat - loss) - 0.05 * (t_avg - t_set)
    return float(t_avg + dt * dtemp)


def plate_setpoint(t: float, t_final: float) -> float:
    """Piecewise setpoint from Problem (30)."""
    if t < 0.5:
        return 58.0
    if t < 1.5:
        return 55.0
    return 63.0 if t <= t_final else 55.0
