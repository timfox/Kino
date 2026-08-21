"""GNN-driven Metropolis MC stub (Methods)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.pgn_nequip.classical_dft import energy_per_particle_kbt
from ltx_trainer.pgn_nequip.config import PGNDesign, PGNNequIPConfig
from ltx_trainer.pgn_nequip.nequip import predict_gnn_energies


def random_configuration(n: int, *, box: float = 10.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, box, size=(n, 3))


def metropolis_step(
    positions: np.ndarray,
    energy_fn,
    *,
    beta: float = 1.0,
    max_disp: float = 0.15,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    n = positions.shape[0]
    i = int(rng.integers(n))
    trial = positions.copy()
    trial[i] += rng.uniform(-max_disp, max_disp, size=3)

    e_old = energy_fn(positions)
    e_new = energy_fn(trial)
    if e_new <= e_old:
        return trial, True
    if rng.random() < np.exp(-beta * (e_new - e_old)):
        return trial, True
    return positions, False


def run_mc_equilibration(
    positions: np.ndarray,
    dft_energies_history: np.ndarray,
    *,
    design: PGNDesign,
    n_steps: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Short MC using GNN surrogate trained on high-energy snapshots."""
    rng = np.random.default_rng(seed)
    pos = positions.copy()

    def gnn_energy(p: np.ndarray) -> float:
        e = energy_per_particle_kbt(p, design=design)
        pred = predict_gnn_energies(np.array([e, e + 0.5]), mae_over_sigma=0.011, seed=seed)
        return float(pred[0])

    accepts = 0
    for _ in range(n_steps):
        pos, ok = metropolis_step(pos, gnn_energy, rng=rng)
        accepts += int(ok)

    e_eq = gnn_energy(pos)
    e_train_mean = float(np.mean(dft_energies_history))
    return {
        "acceptance_ratio": accepts / max(n_steps, 1),
        "equilibrium_energy_kbt": e_eq,
        "training_mean_kbt": e_train_mean,
        "ood_gap_kbt": e_eq - e_train_mean,
    }
