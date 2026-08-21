"""Paper anchors for PGN NequIP study (arXiv:2605.30822)."""

from __future__ import annotations

from typing import Any


def knowledge_card() -> dict[str, Any]:
    return {
        "title": "Using graph neural networks to predict many-body interactions in amorphous materials",
        "authors": "Ghomsheh, Koch, Hormozi (Cornell)",
        "arxiv": "2605.30822",
        "system": "Solvent-free polymer-grafted nanoparticles (soft glass)",
        "method": "Classical DFT → NequIP equivariant GNN → MC equilibration",
        "n_particles": 100,
        "train_val_test": (4000, 500, 500),
        "optimal_nequip": {"lmax": 3, "nl": 3, "rcut_over_d": 6.0},
    }


def fig4_parity_mae() -> list[dict[str, float]]:
    return [
        {"phi_c": 0.1, "sigma_g": 1.8, "mae_over_sigma": 0.011},
        {"phi_c": 0.1, "sigma_g": 1.0, "mae_over_sigma": 0.010},
        {"phi_c": 0.2, "sigma_g": 0.8, "mae_over_sigma": 0.010},
        {"phi_c": 0.3, "sigma_g": 0.5, "mae_over_sigma": 0.009},
        {"phi_c": 0.4, "sigma_g": 0.3, "mae_over_sigma": 0.015},
    ]


def fig5d_nearest_neighbor_experiment() -> list[dict[str, float]]:
    """Normalized nearest-neighbor distance r_nn/d vs φ_c (Fig. 5d)."""
    return [
        {"phi_c": 0.10, "rnn_over_d_gnn": 1.55, "rnn_over_d_exp": 1.52},
        {"phi_c": 0.15, "rnn_over_d_gnn": 1.48, "rnn_over_d_exp": 1.45},
        {"phi_c": 0.20, "rnn_over_d_gnn": 1.42, "rnn_over_d_exp": 1.40},
        {"phi_c": 0.25, "rnn_over_d_gnn": 1.38, "rnn_over_d_exp": 1.36},
        {"phi_c": 0.30, "rnn_over_d_gnn": 1.34, "rnn_over_d_exp": 1.33},
        {"phi_c": 0.35, "rnn_over_d_gnn": 1.31, "rnn_over_d_exp": 1.30},
        {"phi_c": 0.40, "rnn_over_d_gnn": 1.28, "rnn_over_d_exp": 1.27},
    ]


def fig5_hyperuniformity_s0() -> float:
    """Monodisperse limit S(q→0) → 0.5 (Fig. 5c)."""
    return 0.5


def fig6_icosahedral_w6() -> float:
    """Reference ˆw6 for perfect icosahedron (Fig. 6c dashed line)."""
    return 0.0


def cost_speedup() -> dict[str, float]:
    return {"dft_relative": 1.0, "gnn_relative": 1e-4, "speedup": 1e4}
