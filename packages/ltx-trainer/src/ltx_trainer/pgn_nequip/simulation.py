"""Synthetic PGN pipelines: HS sampling → DFT → NequIP → MC."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.pgn_nequip.classical_dft import energy_per_particle_kbt, energy_distribution_stats
from ltx_trainer.pgn_nequip.config import PGNNequIPConfig
from ltx_trainer.pgn_nequip.mc import random_configuration, run_mc_equilibration
from ltx_trainer.pgn_nequip.nequip import mae_over_sigma, predict_gnn_energies, rcut_sweep_mae
from ltx_trainer.pgn_nequip.structure import (
    nearest_neighbor_distance,
    pair_correlation,
    steinhardt_w6_proxy,
    structure_factor_proxy,
)


def sample_hard_sphere_configurations(
    *,
    n_snapshots: int = 64,
    design: PGNNequIPConfig | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = design or PGNNequIPConfig()
    n = cfg.design.n_particles
    positions = []
    energies = []
    for i in range(n_snapshots):
        pos = random_configuration(n, seed=seed + i)
        positions.append(pos)
        energies.append(energy_per_particle_kbt(pos, design=cfg.design))
    return np.stack(positions, axis=0), np.asarray(energies, dtype=np.float64)


def full_pipeline_demo(*, seed: int = 7, cfg: PGNNequIPConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PGNNequIPConfig()
    pos_batch, e_dft = sample_hard_sphere_configurations(n_snapshots=48, design=cfg, seed=seed)
    e_gnn = predict_gnn_energies(e_dft, mae_over_sigma=0.011, seed=seed)
    train_stats = energy_distribution_stats(e_dft)

    pos_eq = pos_batch[0]
    mc = run_mc_equilibration(pos_eq, e_dft, design=cfg.design, n_steps=400, seed=seed + 1)

    r, g = pair_correlation(pos_eq)
    q, sq = structure_factor_proxy(pos_eq, equilibrium=True)
    w6 = steinhardt_w6_proxy(pos_eq)
    rnn = nearest_neighbor_distance(pos_eq)

    return {
        "train_energy_stats": train_stats,
        "mae_over_sigma": mae_over_sigma(e_dft, e_gnn),
        "rcut_sweep": rcut_sweep_mae(),
        "mc": mc,
        "gr_peak": float(g.max()),
        "sq_at_low_q": float(sq[0]),
        "w6_mean": float(np.mean(w6)),
        "rnn_over_d": rnn / 1.0,
        "speedup_vs_dft": cfg.dft_cost_relative / cfg.gnn_cost_relative,
    }
