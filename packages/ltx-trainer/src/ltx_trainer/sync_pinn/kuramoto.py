"""Kuramoto order parameter, dynamics, and toy rollouts (arXiv:2601.00178)."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def shaping_h(t: Array | float, *, horizon: float) -> Array | float:
    """Smooth ramp h(0)=0 used in Eq. (2); linear on [0, T]."""
    if horizon <= 0:
        return 0.0
    return np.minimum(np.asarray(t, dtype=np.float64) / horizon, 1.0)


def full_adjacency(n: int) -> Array:
    a = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(a, 0.0)
    return a


def order_parameter(theta: Array) -> float:
    """Complex order parameter magnitude R in Eq. (6)."""
    z = np.mean(np.exp(1j * np.asarray(theta, dtype=np.float64)))
    return float(np.abs(z))


def kuramoto_rhs(
    theta: Array,
    omega: Array,
    adj: Array,
    u: Array,
    *,
    k: float,
) -> Array:
    """Uncontrolled/controlled Kuramoto RHS Eq. (5)."""
    theta = np.asarray(theta, dtype=np.float64)
    n = theta.size
    coupling = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                coupling[i] += np.sin(theta[j] - theta[i])
    return omega + k * coupling + np.asarray(u, dtype=np.float64)


def sakaguchi_rhs(
    theta: Array,
    omega: Array,
    adj: Array,
    u: Array,
    *,
    k: float,
    alpha: float,
) -> Array:
    """Kuramoto–Sakaguchi Eq. (18)."""
    theta = np.asarray(theta, dtype=np.float64)
    n = theta.size
    coupling = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                coupling[i] += np.sin(theta[j] - theta[i] - alpha)
    return omega + k * coupling + np.asarray(u, dtype=np.float64)


def euler_rollout(
    theta0: Array,
    omega: Array,
    adj: Array,
    u_fn,
    *,
    k: float,
    t_end: float,
    dt: float,
    rhs=kuramoto_rhs,
    alpha: float = 0.0,
) -> tuple[Array, Array, Array]:
    """Simple explicit Euler; returns t, theta_traj (T×N), R_traj."""
    steps = max(2, int(np.ceil(t_end / dt)))
    t = np.linspace(0.0, t_end, steps)
    theta = np.asarray(theta0, dtype=np.float64).copy()
    traj = np.zeros((steps, theta.size), dtype=np.float64)
    r_hist = np.zeros(steps, dtype=np.float64)
    for i, ti in enumerate(t):
        traj[i] = theta
        r_hist[i] = order_parameter(theta)
        if i + 1 >= steps:
            break
        u = np.asarray(u_fn(ti), dtype=np.float64)
        if rhs is sakaguchi_rhs:
            d = rhs(theta, omega, adj, u, k=k, alpha=alpha)
        else:
            d = rhs(theta, omega, adj, u, k=k)
        theta = theta + dt * d
    return t, traj, r_hist


def mean_phase(theta: Array) -> float:
    z = np.mean(np.exp(1j * np.asarray(theta, dtype=np.float64)))
    return float(np.angle(z))


def phase_feedback(theta: Array, *, gain: float, nonlinear: bool) -> Array:
    """Eqs. (15)–(16)."""
    th_bar = mean_phase(theta)
    if nonlinear:
        return -gain * np.sin(np.asarray(theta, dtype=np.float64) - th_bar)
    return -gain * (np.asarray(theta, dtype=np.float64) - th_bar)


def frequency_compensation(omega: Array, *, gain: float) -> Array:
    """Eq. (17) — constant in time for each oscillator."""
    om = np.asarray(omega, dtype=np.float64)
    return -gain * (om - np.mean(om))
