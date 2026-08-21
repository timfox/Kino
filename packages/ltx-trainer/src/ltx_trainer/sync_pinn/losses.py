"""PINN composite loss terms (Eqs. 4, 9–12)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.sync_pinn.kuramoto import kuramoto_rhs, order_parameter


def initial_condition_loss(theta_at_0: np.ndarray, theta0: np.ndarray) -> float:
    """Lic Eq. (11)."""
    d = np.asarray(theta_at_0, dtype=np.float64) - np.asarray(theta0, dtype=np.float64)
    return float(np.mean(d**2))


def dynamics_residual_norm(
    theta: np.ndarray,
    theta_dot: np.ndarray,
    omega: np.ndarray,
    adj: np.ndarray,
    u: np.ndarray,
    *,
    k: float,
) -> float:
    """Scalar ‖r‖² averaged over oscillators for one collocation time (Eq. 9)."""
    r = theta_dot - kuramoto_rhs(theta, omega, adj, u, k=k)
    return float(np.mean(r**2))


def persistence_control_loss(
    t: np.ndarray,
    r_traj: np.ndarray,
    *,
    r_star: float,
    t_star: float,
) -> float:
    """Lcontrol Eq. (10) — mean squared shortfall for t ≥ t*."""
    t = np.asarray(t, dtype=np.float64)
    r_traj = np.asarray(r_traj, dtype=np.float64)
    mask = t >= t_star
    if not np.any(mask):
        return 0.0
    shortfall = np.maximum(0.0, r_star - r_traj[mask])
    return float(np.mean(shortfall**2))


def regulation_loss(u: np.ndarray) -> float:
    """Lreg Eq. (12)."""
    u = np.asarray(u, dtype=np.float64)
    return float(np.mean(u**2))


def composite_training_loss(
    l_dyn: float,
    l_ic: float,
    l_control: float,
    l_reg: float,
) -> float:
    """L = Ldyn + Lic + Lcontrol + Lreg (Eq. 4)."""
    return float(l_dyn + l_ic + l_control + l_reg)
