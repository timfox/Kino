"""Framework card and evaluation demo (arXiv:2605.30134)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.lpm_mcmc.complexity import fast_sweep_cost, faster_sweep_cost, naive_sweep_cost
from ltx_trainer.lpm_mcmc.config import LPMMCMCConfig
from ltx_trainer.lpm_mcmc.link import exact_log_likelihood
from ltx_trainer.lpm_mcmc.mcmc import MCMCState, run_embedding_sweep
from ltx_trainer.lpm_mcmc.metrics import complexity_comparison, empirical_mse_posterior_mean
from ltx_trainer.lpm_mcmc.moments import MomentStore
from ltx_trainer.lpm_mcmc.partition import block_partition, is_b_good
from ltx_trainer.lpm_mcmc.taylor import approximate_log_likelihood, taylor_error_bound, tv_error_bound


def framework_card(cfg: LPMMCMCConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LPMMCMCConfig()
    return {
        "name": "Fast LPM MCMC",
        "paper": cfg.paper_arxiv,
        "task": "Accurate efficient MCMC for latent position network models",
        "algorithms": {
            "Algorithm9": "Moment sketch + κ-order Taylor (fast)",
            "Algorithm10": "First-order moments only (faster)",
        },
        "data_structures": ["M_{s,t,α}", "M^{(c)}_{s,t,α}", "Q_{i,t,α}", "Q^{(c)}_{t,α}"],
        "link": "Gaussian LPM — Eq. (7.1)",
        "baseline": "RMF24 grid noisy MwG",
        "complexity": {
            "fast": "O(|E||A| + nK|A|²)",
            "faster": "O(nK)",
            "naive_MwG": "O(n²) per sweep",
        },
    }


def paper_limitations() -> list[str]:
    return [
        "Analysis restricted to 2D compact Ω; high-dimensional LPMs differ asymptotically.",
        "TV error bounds can exceed 1 for realistic b unless κ and b scale with n.",
        "Partition quality assumes b-good embedding (Assumption 6.8).",
    ]


def benchmark_manifest(cfg: LPMMCMCConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LPMMCMCConfig()
    return {
        "simulation": "Gaussian link β0=0.1, β1=0.7, σ=0.6, τ ~ truncated Gaussian on [0,1]²",
        "studies": ["Rainbow posterior mean (Fig 1–2)", "Single-node contour (Fig 4)", "Distance PDF (Fig 5–6)"],
        "sizes": [2000, 30000],
        "block_widths": [0.125, 0.0625, 0.03125, 0.1, 0.05],
        "orders": {"fast": cfg.taylor_order, "faster": 1},
    }


def evaluation_demo(*, cfg: LPMMCMCConfig | None = None, n: int = 24, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or LPMMCMCConfig()
    rng = np.random.default_rng(seed)
    tau = rng.uniform(0.0, 1.0, size=(n, 2))

    edges: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if rng.uniform() < 0.25:
                edges.add((i, j))

    partition, k = block_partition(tau, block_width=cfg.block_width)
    store = MomentStore.build(tau, edges, partition, kappa=min(cfg.taylor_order, 1))
    centers = {s: tau[partition == s].mean(axis=0) for s in range(k)}

    l_exact = exact_log_likelihood(tau, edges, cfg=cfg)
    l_approx = approximate_log_likelihood(store, centers, cfg=cfg)

    state = MCMCState(
        tau=tau,
        partition=partition,
        edges=edges,
        store=store,
        approx_loglik=l_approx,
        cfg=cfg,
    )
    accepts, state = run_embedding_sweep(state, rng=rng)

    r_bound = taylor_error_bound(n, kappa=cfg.taylor_order, block_width=cfg.block_width, mg=cfg.mg, bg=cfg.bg)
    tv_bound = tv_error_bound(
        n,
        kappa=cfg.taylor_order,
        block_width=cfg.block_width,
        mg=cfg.mg,
        bg=cfg.bg,
        epsilon_post=cfg.epsilon_post,
    )

    num_edges = len(edges)
    costs = complexity_comparison(n, num_edges, k, cfg.taylor_order)

    return {
        "n_vertices": n,
        "n_edges": num_edges,
        "n_blocks": k,
        "partition_b_good": is_b_good(tau, partition, cfg.block_width),
        "loglik_exact": round(l_exact, 4),
        "loglik_approx": round(l_approx, 4),
        "taylor_error_bound_R": round(r_bound, 6),
        "tv_error_bound": round(tv_bound, 6),
        "sweep_accepts": accepts,
        "cost_fast": fast_sweep_cost(n, num_edges, k, cfg.taylor_order),
        "cost_faster": faster_sweep_cost(n, k),
        "cost_naive": naive_sweep_cost(n),
        "speedup_vs_naive": round(naive_sweep_cost(n) / max(faster_sweep_cost(n, k), 1), 2),
        "complexity_table": costs,
    }


def training_step_demo(*, cfg: LPMMCMCConfig | None = None) -> dict[str, Any]:
    return evaluation_demo(cfg=cfg)
