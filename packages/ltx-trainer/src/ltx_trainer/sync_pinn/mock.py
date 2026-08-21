"""Toy PINN-control smoke without neural training (arXiv:2601.00178)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.sync_pinn.config import SyncPinnConfig
from ltx_trainer.sync_pinn.kuramoto import (
    euler_rollout,
    frequency_compensation,
    full_adjacency,
    kuramoto_rhs,
    order_parameter,
    phase_feedback,
    sakaguchi_rhs,
)
from ltx_trainer.sync_pinn.losses import (
    composite_training_loss,
    dynamics_residual_norm,
    initial_condition_loss,
    persistence_control_loss,
    regulation_loss,
)
from ltx_trainer.sync_pinn.metrics import (
    integrated_control_cost,
    instantaneous_control_cost,
    persistence_satisfied,
    relative_sync_error,
    time_averaged_order,
)


def _demo_control_profile(t: float, *, t_star: float, n: int) -> np.ndarray:
    """Smooth ramp-down control mimicking learned PINN profiles (Fig. 3)."""
    if t < 0:
        return np.zeros(n)
    amp = 0.35 * np.exp(-0.6 * max(0.0, t - 0.2 * t_star))
    return amp * np.sin(np.linspace(0.0, 2.0 * np.pi, n))


def evaluation_smoke(cfg: SyncPinnConfig | None = None) -> dict[str, Any]:
    c = cfg or SyncPinnConfig()
    rng = np.random.default_rng(7)
    n = c.n_oscillators
    adj = full_adjacency(n)
    omega = rng.uniform(-np.pi, np.pi, size=n)
    theta0 = rng.uniform(-np.pi / 2, np.pi / 2, size=n)

    dt = 0.02
    u_fn = lambda t: _demo_control_profile(t, t_star=c.t_target, n=n)
    t, _, r_ctrl = euler_rollout(
        theta0, omega, adj, u_fn, k=c.coupling_k, t_end=c.control_horizon, dt=dt
    )
    _, _, r_free = euler_rollout(
        theta0,
        omega,
        adj,
        lambda _t: np.zeros(n),
        k=c.coupling_k,
        t_end=c.control_horizon,
        dt=dt,
    )

    u0 = u_fn(0.0)
    l_ic = initial_condition_loss(theta0, theta0)
    l_dyn = dynamics_residual_norm(
        theta0,
        kuramoto_rhs(theta0, omega, adj, u0, k=c.coupling_k),
        omega,
        adj,
        u0,
        k=c.coupling_k,
    )
    l_control = persistence_control_loss(t, r_ctrl, r_star=c.r_target, t_star=c.t_target)
    l_reg = regulation_loss(u0)
    l_total = composite_training_loss(l_dyn, l_ic, l_control, l_reg)

    p_series = np.array([instantaneous_control_cost(u_fn(ti)) for ti in t])
    e_ctrl = integrated_control_cost(t, p_series)

    # Phase feedback rollout (state-dependent u)
    steps = len(t)
    theta_ph = theta0.copy()
    p_ph = np.zeros(steps)
    for i, ti in enumerate(t):
        u_ph = phase_feedback(theta_ph, gain=c.phase_feedback_gain, nonlinear=False)
        p_ph[i] = instantaneous_control_cost(u_ph)
        if i + 1 >= steps:
            break
        theta_ph = theta_ph + dt * kuramoto_rhs(theta_ph, omega, adj, u_ph, k=c.coupling_k)
    e_ph = integrated_control_cost(t, p_ph)

    u_freq = frequency_compensation(omega, gain=c.freq_comp_gain)
    t_fc, _, r_fc = euler_rollout(
        theta0,
        omega,
        adj,
        lambda _t: u_freq,
        k=c.coupling_k,
        t_end=c.control_horizon,
        dt=dt,
    )
    e_fc = integrated_control_cost(
        t_fc, np.full_like(t_fc, instantaneous_control_cost(u_freq))
    )

    # Sakaguchi frustrated
    t_sk, _, r_sk = euler_rollout(
        theta0,
        omega,
        adj,
        u_fn,
        k=c.sakaguchi_k,
        t_end=c.control_horizon,
        dt=dt,
        rhs=sakaguchi_rhs,
        alpha=c.sakaguchi_alpha,
    )

    r_mean = time_averaged_order(t, r_ctrl, t_star=c.t_target)
    return {
        "paper": c.paper_arxiv,
        "n_oscillators": n,
        "R_free_final": round(float(r_free[-1]), 4),
        "R_ctrl_final": round(float(r_ctrl[-1]), 4),
        "persistence_ok": persistence_satisfied(
            t, r_ctrl, r_star=c.r_target, t_star=c.t_target
        ),
        "loss_total": round(l_total, 6),
        "loss_control": round(l_control, 6),
        "integrated_cost_pinn_toy": round(e_ctrl, 4),
        "integrated_cost_phase_feedback": round(e_ph, 4),
        "integrated_cost_freq_comp": round(e_fc, 4),
        "order_parameter_at_t_star": round(order_parameter(theta0), 4),
        "R_mean_after_t_star": round(r_mean, 4),
        "relative_sync_error": round(relative_sync_error(r_mean, c.r_target), 4),
        "sakaguchi_R_final": round(float(r_sk[-1]), 4),
        "pinn_smooth_peak_P": round(float(np.max(p_series[: int(0.5 * len(p_series))])), 4),
        "phase_peak_P_early": round(float(np.max(p_ph[: max(1, int(0.3 * len(p_ph)))])), 4),
    }
