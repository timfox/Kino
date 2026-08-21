"""Control cost and synchronization metrics (Eqs. 7, 13–14)."""

from __future__ import annotations

import numpy as np


def instantaneous_control_cost(u: np.ndarray) -> float:
    """P(t) Eq. (13)."""
    u = np.asarray(u, dtype=np.float64)
    return float(np.mean(u**2))


def integrated_control_cost(t: np.ndarray, p: np.ndarray) -> float:
    """E = ∫ P(t) dt via trapezoid rule (Eq. 14)."""
    return float(np.trapezoid(np.asarray(p, dtype=np.float64), np.asarray(t, dtype=np.float64)))


def persistence_satisfied(
    t: np.ndarray,
    r: np.ndarray,
    *,
    r_star: float,
    t_star: float,
) -> bool:
    """Check R(t) ≥ R* for all t ≥ t* (design constraint behind Eq. 8)."""
    t = np.asarray(t, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    mask = t >= t_star
    if not np.any(mask):
        return True
    return bool(np.all(r[mask] >= r_star - 1e-9))


def time_averaged_order(
    t: np.ndarray,
    r: np.ndarray,
    *,
    t_star: float,
) -> float:
    """⟨R⟩ over [t*, T] used in noise robustness Sec. IV D."""
    t = np.asarray(t, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    mask = t >= t_star
    if not np.any(mask):
        return float(np.mean(r))
    return float(np.mean(r[mask]))


def relative_sync_error(r_mean: float, r_star: float) -> float:
    """ER = (R* − ⟨R⟩) / R* (Fig. 8)."""
    if r_star <= 0:
        return 0.0
    return float(max(0.0, (r_star - r_mean) / r_star))
