"""Paper anchors for SLE κ neural networks (arXiv:2606.02682)."""

from __future__ import annotations

from typing import Any


def knowledge_card() -> dict[str, Any]:
    return {
        "title": "Neural Networks and Schramm-Loewner Evolutions",
        "authors": "Neilesh Shrotri, Vlad Margarint",
        "arxiv": "2606.02682",
        "task": "Predict Loewner driver c and SLEκ parameter κ from trajectories/traces",
        "datasets": {"trajectories": 10_000, "train_fraction": 0.8, "test_fraction": 0.2},
        "physics_anchors": {
            "lerw": 2.0,
            "ising": 3.0,
            "percolation": 6.0,
        },
    }


def test_losses() -> dict[str, float]:
    return {
        "deterministic_c_mse": 0.00264,
        "sle_kappa_same_noise_mse": 0.345,
        "sle_kappa_different_noise_mse": 3.98,
        "sle_trace_fixed_brownian_mse": 0.194,
    }


def sle_regimes() -> list[dict[str, Any]]:
    """Three statistical behaviors as κ varies on [0, ∞)."""
    return [
        {"regime": "κ < 4", "behavior": "simple curves, no self-intersection"},
        {"regime": "κ = 4", "behavior": "critical self-avoiding interface"},
        {"regime": "κ > 4", "behavior": "space-filling / rough curves"},
    ]


def simulation_hyperparams() -> dict[str, Any]:
    return {
        "c_uniform": [0.0, 3.0],
        "kappa_uniform": [0.0, 8.0],
        "z0_uniform": [0.0, 10.0],
        "t_start": 0.1,
        "t_end": 1.0,
        "n_steps": 100,
    }
