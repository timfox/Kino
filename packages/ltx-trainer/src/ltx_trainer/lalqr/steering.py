"""LA-LQR closed-loop steering and synthetic dynamics smoke."""

from __future__ import annotations

import numpy as np

from ltx_trainer.lalqr.config import LalqrConfig, LQRWeights
from ltx_trainer.lalqr.lqr import la_lqr_text_control, solve_ltv_lqr
from ltx_trainer.lalqr.setpoints import (
    latent_feature_direction,
    latent_feature_strength,
    llfs_setpoint,
    tracking_error,
)
from ltx_trainer.lalqr.subspace import (
    captured_energy_fraction,
    contrastive_rows,
    mean_contrastive_vector,
    project_activation,
    randomized_svd_basis,
)


def synthetic_local_dynamics(
    *,
    horizon: int,
    d_lat: int,
    d_text: int,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Stable random LTV model for CPU Riccati smoke (not fitted Jacobians)."""
    rng = np.random.default_rng(seed)
    a_list: list[np.ndarray] = []
    b_list: list[np.ndarray] = []
    for _ in range(horizon):
        m = rng.standard_normal((d_lat, d_lat)) * 0.05
        a = 0.92 * np.eye(d_lat) + m
        b = rng.standard_normal((d_lat, d_text)) * 0.01
        a_list.append(a)
        b_list.append(b)
    return a_list, b_list


def build_latent_controller(
    basis: np.ndarray,
    contrast_rows: np.ndarray,
    a_list: list[np.ndarray],
    b_list: list[np.ndarray],
    weights: LQRWeights,
) -> dict[str, np.ndarray | float | list[np.ndarray]]:
    """Subspace + LLFS + LQR gains for one layer partition."""
    ez, vz = latent_feature_direction(basis, contrast_rows)
    mu = mean_contrastive_vector(contrast_rows)
    rho = captured_energy_fraction(basis, mu)
    beta_star = llfs_setpoint(ez, lambda_setpoint=weights.lambda_setpoint)
    gains = solve_ltv_lqr(a_list, b_list, q=weights.q, r=weights.r_text, q_terminal=weights.q_terminal)
    return {
        "basis": basis,
        "ez": ez,
        "vz": vz,
        "rho": rho,
        "beta_star": beta_star,
        "gains": gains,
    }


def steer_latent_chain(
    controller: dict[str, np.ndarray | float | list[np.ndarray]],
    activation: np.ndarray,
    *,
    gain_index: int = 0,
    u_bar: np.ndarray | None = None,
) -> dict[str, float | np.ndarray]:
    """One LA-LQR text-embedding correction from a vectorized activation."""
    basis = controller["basis"]
    assert isinstance(basis, np.ndarray)
    vz = controller["vz"]
    assert isinstance(vz, np.ndarray)
    beta_star = controller["beta_star"]
    assert isinstance(beta_star, float)
    gains = controller["gains"]
    assert isinstance(gains, list)

    z = project_activation(basis, activation)
    alpha = tracking_error(z, vz, beta_star)
    beta_z = latent_feature_strength(z, vz)
    k = gains[min(gain_index, len(gains) - 1)]
    u = la_lqr_text_control(k, vz, alpha, u_bar=u_bar)
    return {
        "z": z,
        "alpha": alpha,
        "beta_z": beta_z,
        "beta_star": beta_star,
        "control": u,
        "control_norm": float(np.linalg.norm(u)),
    }


def synthesize_contrastive_activations(
    cfg: LalqrConfig,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Random paired activations for stub demos."""
    rng = np.random.default_rng(seed)
    d = cfg.demo_activation_dim
    n = cfg.contrastive_pairs
    direction = rng.standard_normal(d)
    direction /= np.linalg.norm(direction) + 1e-8
    pos = rng.standard_normal((n, d)) * 0.2 + 0.8 * direction
    neg = rng.standard_normal((n, d)) * 0.2 - 0.1 * direction
    return pos, neg


def run_lalqr_smoke(
    cfg: LalqrConfig | None = None,
    *,
    category: str = "Pornography",
    model: str = "Wan2.1-T2V-14B LightX2V",
    seed: int = 42,
) -> dict[str, float | int | str]:
    cfg = cfg or LalqrConfig()
    weights = cfg.lqr_for(model=model, category=category)
    pos, neg = synthesize_contrastive_activations(cfg, seed=seed)
    rows = contrastive_rows(pos, neg)
    basis = randomized_svd_basis(
        rows,
        rank=cfg.latent_rank,
        oversampling=cfg.svd_oversampling,
        seed=cfg.svd_seed + seed,
    )
    a_list, b_list = synthetic_local_dynamics(
        horizon=cfg.horizon_layers,
        d_lat=basis.shape[1],
        d_text=min(cfg.text_dim, 128),
        seed=seed + 7,
    )
    # Truncate B to demo text dim
    b_list = [b[:, : min(b.shape[1], 64)] for b in b_list]
    ctrl = build_latent_controller(basis, rows, a_list, b_list, weights)
    probe = pos[0]
    step = steer_latent_chain(ctrl, probe, gain_index=0)
    return {
        "category": category,
        "model": model,
        "latent_rank": int(basis.shape[1]),
        "rho": float(ctrl["rho"]),
        "alpha": float(step["alpha"]),
        "beta_z": float(step["beta_z"]),
        "beta_star": float(step["beta_star"]),
        "control_norm": float(step["control_norm"]),
        "q": weights.q,
        "r_text": weights.r_text,
        "lambda_setpoint": weights.lambda_setpoint,
    }
