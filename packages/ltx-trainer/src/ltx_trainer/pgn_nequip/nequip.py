"""NequIP surrogate and hyperparameter analysis stubs (Fig. 4)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.pgn_nequip.config import PGNNequIPConfig


def normalize_energies(energies: np.ndarray) -> tuple[np.ndarray, float, float]:
    mu = float(np.mean(energies))
    sigma = float(np.std(energies)) or 1.0
    return (energies - mu) / sigma, mu, sigma


def predict_gnn_energies(
    dft_energies: np.ndarray,
    *,
    mae_over_sigma: float = 0.011,
    seed: int = 0,
) -> np.ndarray:
    """Parity-style GNN predictions with controlled MAE/σ_E."""
    rng = np.random.default_rng(seed)
    norm, mu, sigma = normalize_energies(dft_energies)
    noise = rng.normal(scale=mae_over_sigma, size=norm.shape)
    return mu + sigma * (norm + noise)


def mae_over_sigma(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _, mu, sigma = normalize_energies(y_true)
    return float(np.mean(np.abs(y_pred - y_true)) / sigma)


def rcut_sweep_mae() -> list[dict[str, float]]:
    """Fig. 4b — MAE/σ_E vs rcut/d (σ_g=1.8, φ_c=0.1)."""
    rcut = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    # Order-of-magnitude drop then saturation past rcut/d=6.
    mae = np.array([0.42, 0.18, 0.07, 0.03, 0.011, 0.010, 0.010])
    return [{"rcut_over_d": float(r), "mae_over_sigma": float(m)} for r, m in zip(rcut, mae)]


def nl_lmax_sweep() -> list[dict[str, Any]]:
    """Fig. 4c — MAE vs message-passing layers and lmax."""
    rows: list[dict[str, Any]] = []
    base = {0: 0.35, 1: 0.12, 2: 0.045, 3: 0.025, 4: 0.024, 5: 0.024}
    for lmax in (0, 1, 2, 3):
        scale = {0: 1.0, 1: 0.55, 2: 0.35, 3: 0.25}[lmax]
        for nl in range(1, 6):
            rows.append(
                {
                    "lmax": lmax,
                    "nl": nl,
                    "mae_over_sigma": float(base[nl] * scale),
                }
            )
    return rows


def learning_curve_exponents() -> dict[int, float]:
    """Fig. 4d power-law ε ~ N_t^{-a}."""
    return {0: 0.463, 1: 0.587, 2: 0.676, 3: 0.720}


def learning_curve_mae(nt: int, lmax: int) -> float:
    a = learning_curve_exponents()[lmax]
    return 0.35 * (nt ** (-a))


def parity_samples(cfg: PGNNequIPConfig | None = None) -> list[dict[str, float]]:
    """Fig. 4a reported MAE/σ_E across design parameters."""
    cfg = cfg or PGNNequIPConfig()
    return [
        {"phi_c": 0.1, "sigma_g": 1.8, "mw_kda": 5.0, "mae_over_sigma": 0.011},
        {"phi_c": 0.1, "sigma_g": 1.0, "mw_kda": 9.0, "mae_over_sigma": 0.010},
        {"phi_c": 0.2, "sigma_g": 0.8, "mw_kda": 5.0, "mae_over_sigma": 0.010},
        {"phi_c": 0.3, "sigma_g": 0.5, "mw_kda": 5.0, "mae_over_sigma": 0.009},
        {"phi_c": 0.4, "sigma_g": 0.3, "mw_kda": 5.0, "mae_over_sigma": 0.015},
    ]
