"""CFM-based mel refinement (Eq. 12–22)."""

from __future__ import annotations

import numpy as np


def linear_path(m0: np.ndarray, target: np.ndarray, t: float) -> np.ndarray:
    """Eq. (15): M_t = (1-t) M0 + t M."""
    return (1.0 - t) * m0 + t * target


def path_velocity(target: np.ndarray, m0: np.ndarray) -> np.ndarray:
    """Eq. (16): dM/dt = M - M0 along OT-CFM linear path."""
    return target - m0


def cfm_loss(velocity_pred: np.ndarray, target: np.ndarray, m0: np.ndarray) -> float:
    """Eq. (17): ||v_θ - (M - M0)||^2."""
    target_v = path_velocity(target, m0)
    diff = velocity_pred - target_v
    return float(np.mean(diff ** 2))


def euler_step(state: np.ndarray, velocity: np.ndarray, dt: float) -> np.ndarray:
    """Eq. (14): explicit Euler ODE integration."""
    return state + dt * velocity


def ideal_terminal_operator(state: np.ndarray, t: float, velocity: np.ndarray) -> np.ndarray:
    """Eq. (19): ITO = M_t + (1-t) v_θ."""
    return state + (1.0 - t) * velocity


def self_consistency_loss(
    v_t: np.ndarray,
    v_t_dt: np.ndarray,
) -> float:
    """Eq. (21): encourage time-invariant velocity field."""
    diff = v_t - v_t_dt
    return float(np.mean(diff ** 2))


def refinement_loss(
    v_t: np.ndarray,
    v_t_dt: np.ndarray,
    target: np.ndarray,
    m0: np.ndarray,
    *,
    lambda_cfm: float = 45.0,
    lambda_self_cons: float = 10.0,
    use_self_cons: bool = True,
) -> float:
    """Eq. (22): L_ref = λ_CFM L_CFM + λ_self L_self-cons."""
    l_cfm = cfm_loss(v_t, target, m0)
    if not use_self_cons:
        return lambda_cfm * l_cfm
    l_sc = self_consistency_loss(v_t, v_t_dt)
    return lambda_cfm * l_cfm + lambda_self_cons * l_sc


def refine_mel_euler(
    m0: np.ndarray,
    velocity_fn,
    *,
    steps: int = 4,
    conditioning: np.ndarray | None = None,
) -> np.ndarray:
    """Integrate Eq. (13–14) with I Euler steps."""
    state = m0.copy()
    dt = 1.0 / steps
    for i in range(steps):
        t = i * dt
        v = velocity_fn(state, t, conditioning)
        state = euler_step(state, v, dt)
    return state
