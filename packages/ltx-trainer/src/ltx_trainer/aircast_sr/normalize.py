"""Min–max and log-precip normalization for AORC / GraphCast fields."""

from __future__ import annotations

import numpy as np

from ltx_trainer.aircast_sr.variables import TARGET_VARIABLES


def log_precip_transform(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0))


def minmax_normalize(
    x: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    log_precip: bool = False,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if log_precip:
        arr = log_precip_transform(arr)
    denom = max(vmax - vmin, 1e-8)
    return np.clip((arr - vmin) / denom, 0.0, 1.0)


def minmax_denormalize(
    x: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    log_precip: bool = False,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64) * (vmax - vmin) + vmin
    if log_precip:
        return np.expm1(arr)
    return arr


def default_variable_bounds() -> dict[str, tuple[float, float]]:
    """Toy bounds for smoke demos (not operational AORC stats)."""
    return {
        "precipitation": (0.0, 50.0),
        "t2m": (250.0, 310.0),
        "q2m": (0.0, 0.025),
        "u10": (-25.0, 25.0),
        "v10": (-25.0, 25.0),
        "sp": (85000.0, 105000.0),
        "dlwrf": (150.0, 450.0),
    }


def normalize_target_stack(stack: np.ndarray, bounds: dict[str, tuple[float, float]] | None = None) -> np.ndarray:
    """stack shape (n_var, ...) in physical units → [0, 1]."""
    bounds = bounds or default_variable_bounds()
    out = np.zeros_like(stack, dtype=np.float64)
    for i, name in enumerate(TARGET_VARIABLES):
        vmin, vmax = bounds[name]
        out[i] = minmax_normalize(stack[i], vmin=vmin, vmax=vmax, log_precip=(name == "precipitation"))
    return out
