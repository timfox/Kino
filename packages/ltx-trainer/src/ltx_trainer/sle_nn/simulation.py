"""Synthetic Loewner / SLEκ → NN prediction demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.sle_nn.config import SleNNConfig
from ltx_trainer.sle_nn.loewner_deterministic import sample_deterministic_trajectory
from ltx_trainer.sle_nn.loewner_sle import generate_sle_dataset
from ltx_trainer.sle_nn.neural_net import (
    architecture_summary,
    deterministic_c_predictions,
    mse,
    sle_kappa_predictions,
    sle_trace_kappa_predictions,
)


def _train_test_split(
    values: np.ndarray,
    *,
    train_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = values.size
    idx = rng.permutation(n)
    split = int(train_fraction * n)
    train_idx, test_idx = idx[:split], idx[split:]
    return values[train_idx], values[test_idx]


def full_pipeline_demo(*, seed: int = 0, cfg: SleNNConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SleNNConfig()
    sim = cfg.sim
    rng = np.random.default_rng(seed)

    n = 256
    c_vals = rng.uniform(sim.c_min, sim.c_max, size=n)
    z0 = complex(rng.uniform(0, 10), rng.uniform(0, 10))
    traj = sample_deterministic_trajectory(
        float(c_vals[0]),
        z0,
        t_start=sim.t_start,
        t_end=sim.t_end,
        n_steps=sim.n_steps,
    )

    c_train, c_test = _train_test_split(c_vals, train_fraction=sim.train_fraction, seed=seed)
    c_pred = deterministic_c_predictions(c_test, cfg=cfg, seed=seed + 1)
    det_mse = mse(c_test, c_pred)

    _, kappa_all = generate_sle_dataset(128, seed=seed + 2, shared_brownian=False)
    k_train, k_test = _train_test_split(kappa_all, train_fraction=sim.train_fraction, seed=seed + 3)
    k_same_pred = sle_kappa_predictions(k_test, same_noise=True, cfg=cfg, seed=seed + 4)
    k_diff_pred = sle_kappa_predictions(k_test, same_noise=False, cfg=cfg, seed=seed + 5)
    _, kappa_shared = generate_sle_dataset(128, seed=seed + 6, shared_brownian=True)
    k_trace_train, k_trace_test = _train_test_split(
        kappa_shared, train_fraction=sim.train_fraction, seed=seed + 7
    )
    k_trace_pred = sle_trace_kappa_predictions(k_trace_test, cfg=cfg, seed=seed + 8)

    return {
        "example_trajectory_len": int(traj.size),
        "deterministic_test_mse": det_mse,
        "sle_same_noise_test_mse": mse(k_test, k_same_pred),
        "sle_different_noise_test_mse": mse(k_test, k_diff_pred),
        "sle_trace_fixed_bm_test_mse": mse(k_trace_test, k_trace_pred),
        "architecture": architecture_summary(cfg.nn),
        "train_fraction": sim.train_fraction,
        "n_trajectories_paper": sim.n_trajectories,
    }
