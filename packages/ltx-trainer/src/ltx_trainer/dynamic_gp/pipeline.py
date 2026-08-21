"""Heat-equation DGP demo — Sec. 6.1."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.dynamic_gp.basis import fourier_basis, gram_matrix, transition_matrix
from ltx_trainer.dynamic_gp.benchmarks import summary_anchors
from ltx_trainer.dynamic_gp.config import DynamicGPConfig
from ltx_trainer.dynamic_gp.error_analysis import error_decomposition_stub, functional_l2_error
from ltx_trainer.dynamic_gp.kalman import KalmanState, kalman_predict, kalman_update, mean_on_grid
from ltx_trainer.dynamic_gp.kernels import heat_kernel, project_kernel_to_basis, squared_exponential_cov, wave_gaussian_delta


def _initial_impulse(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    out[np.abs(x) < 0.05] = 10.0
    return out


def _project_to_coeffs(u_grid: np.ndarray, f: np.ndarray, dx: float, lam_u: np.ndarray) -> np.ndarray:
    """L2 projection of f onto span{U} — Eq. (4.8) for D=1."""
    return np.linalg.solve(lam_u, u_grid @ (f * dx))


def run_demo(cfg: DynamicGPConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DynamicGPConfig()
    d = cfg.state_dim
    m = cfg.num_basis
    lo, hi = cfg.domain_lo, cfg.domain_hi
    n = 64
    x = np.linspace(lo, hi, n)
    dx = (hi - lo) / (n - 1)

    u_grid = fourier_basis(x, m, lo, hi)
    lam_u = gram_matrix(u_grid, dx)
    k_mat = heat_kernel(x, x, alpha=cfg.alpha_diffusivity, dt=cfg.dt)
    lam_f = project_kernel_to_basis(k_mat, u_grid, dx, d=d)
    qf = project_kernel_to_basis(squared_exponential_cov(x, x, amp=0.1, length=0.3), u_grid, dx, d=d)
    qv = project_kernel_to_basis(squared_exponential_cov(x, x, amp=0.1, length=0.1), u_grid, dx, d=d)
    a_m = transition_matrix(lam_f, lam_u, d)

    f0 = _initial_impulse(x)
    z_true = _project_to_coeffs(u_grid, f0, dx, lam_u)
    state = KalmanState(z=z_true.copy(), psi=qf.copy())

    errors: list[float] = []
    rng = np.random.default_rng(0)
    obs_idx = rng.choice(n, size=cfg.num_obs, replace=False)

    for _ in range(cfg.num_steps):
        z_true = a_m @ z_true
        f_true = mean_on_grid(u_grid, z_true)
        y = f_true[obs_idx] + rng.normal(0, cfg.sigma_w, cfg.num_obs)
        c = u_grid[:, obs_idx].T
        w = np.eye(cfg.num_obs) * (cfg.sigma_w**2)
        state = kalman_update(state, y, c, w)
        f_hat = mean_on_grid(u_grid, state.z)
        errors.append(functional_l2_error(f_true, f_hat, dx))
        state = kalman_predict(state, a_m, qv)

    decomp = error_decomposition_stub(state.psi, state.psi, lam_u, out_of_subspace=0.05, d=d)
    summary = summary_anchors()
    return {
        "config": {"M": m, "alpha": cfg.alpha_diffusivity, "steps": cfg.num_steps},
        "final_l2_error": round(errors[-1], 4),
        "mean_l2_error": round(float(np.mean(errors)), 4),
        "error_decomposition": decomp,
        "errors_over_steps": [round(e, 4) for e in errors],
        "summary": summary,
    }


def run_wave_demo(cfg: DynamicGPConfig | None = None) -> dict[str, Any]:
    """Wave equation stub — Sec. 6.2 (D=2 state, Gaussian-smoothed kernels)."""
    cfg = cfg or DynamicGPConfig(state_dim=2, num_basis=31, num_steps=10, num_obs=3, sigma_w=0.003)
    from ltx_trainer.dynamic_gp.constants import WAVE_EXAMPLE

    lo, hi = -10.0, 10.0
    n = 128
    x = np.linspace(lo, hi, n)
    dx = (hi - lo) / (n - 1)
    c, dt = WAVE_EXAMPLE["wave_speed_c"], WAVE_EXAMPLE["dt"]
    eps = dx

    m = cfg.num_basis
    u_grid = fourier_basis(x, m, lo, hi)
    lam_u = gram_matrix(u_grid, dx)
    k_phi = wave_gaussian_delta(x, x, c=c, dt=dt, eps=eps)
    lam_f = project_kernel_to_basis(k_phi, u_grid, dx, d=1)
    lam_f2 = np.kron(np.eye(2), lam_f)
    a_m = transition_matrix(lam_f2, lam_u, d=2)

    m0 = 10.0 * np.exp(-(x**2) / 2.0)
    z_true = np.zeros(2 * m)
    z_true[:m] = _project_to_coeffs(u_grid, m0, dx, lam_u)
    state = KalmanState(z=z_true.copy(), psi=np.eye(2 * m) * 0.01)

    obs_idx = np.array([n // 4, n // 2, 3 * n // 4])
    rng = np.random.default_rng(1)

    for _ in range(cfg.num_steps):
        z_true = a_m @ z_true
        phi_true = mean_on_grid(u_grid, z_true[:m])
        y = phi_true[obs_idx] + rng.normal(0, cfg.sigma_w, cfg.num_obs)
        c_obs = np.zeros((cfg.num_obs, 2 * m))
        c_obs[:, :m] = u_grid[:, obs_idx].T
        w = np.eye(cfg.num_obs) * (cfg.sigma_w**2)
        state = kalman_update(state, y, c_obs, w)
        state = kalman_predict(state, a_m, np.zeros((2 * m, 2 * m)))

    f_hat = mean_on_grid(u_grid, state.z[:m])
    err = functional_l2_error(phi_true, f_hat, dx)
    return {
        "wave_speed_c": c,
        "dt": dt,
        "M_fourier": m,
        "D_state": 2,
        "final_l2_error": round(err, 4),
        "anchor": WAVE_EXAMPLE,
    }
