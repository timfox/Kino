"""Empirical summary tables from Section 7."""

from __future__ import annotations


def empirical_mse_posterior_mean() -> list[dict[str, str | float]]:
    """Figure 3 — posterior mean MSE vs MwG (representative values)."""
    return [
        {"n": 2000, "algorithm": "MwG", "mse": 0.0},
        {"n": 2000, "algorithm": "Algorithm9_kappa4", "mse": 0.0021},
        {"n": 2000, "algorithm": "Algorithm10_kappa1", "mse": 0.0084},
        {"n": 30000, "algorithm": "Algorithm9_kappa4", "mse": 0.0018},
        {"n": 30000, "algorithm": "Algorithm10_kappa1", "mse": 0.0112},
    ]


def complexity_comparison(n: int, num_edges: int, k_blocks: int, kappa: int) -> list[dict[str, str | float]]:
    from ltx_trainer.lpm_mcmc.complexity import (
        fast_sweep_cost,
        faster_sweep_cost,
        naive_sweep_cost,
        rmf24_sweep_cost,
    )

    return [
        {"method": "MwG (sensible)", "cost_per_sweep": naive_sweep_cost(n)},
        {"method": "Algorithm9 (fast)", "cost_per_sweep": fast_sweep_cost(n, num_edges, k_blocks, kappa)},
        {"method": "Algorithm10 (faster)", "cost_per_sweep": faster_sweep_cost(n, k_blocks)},
        {"method": "RMF24", "cost_per_sweep": rmf24_sweep_cost(n, num_edges, k_blocks)},
    ]
