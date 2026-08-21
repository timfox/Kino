"""Framework card and paper tables for MOSM+DP."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mosm_dp.algorithms import (
    dp_bicriteria_smoke,
    dp_multi_greedy,
    toy_three_objective_instance,
)
from ltx_trainer.mosm_dp.config import MosmDpConfig


def framework_card(cfg: MosmDpConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MosmDpConfig()
    return {
        "name": "MOSM-DP",
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "objective": "max_S min_j f_j(S) subject to |S|≤k and ε-DP",
        "algorithms": ["DP-MultiGreedy", "DP-Bicriteria"],
        "constraint": "requires d ≤ k for MultiGreedy phase-1 budgets",
        "phases": [
            "Phase 1: DP-Greedy per objective with budget ⌊k/d⌋ and ε₁/d",
            "Phase 2: noisy greedy on minimax F(S) until |S|=k",
        ],
    }


def table_approximation_guarantees() -> list[dict[str, Any]]:
    return [
        {
            "algorithm": "DP-MultiGreedy",
            "approx": "Ω(min_j f_j(S*) · (1 - O(d log k / k)))",
            "privacy": "(ε,δ)-DP",
            "note": "d ≤ k",
        },
        {
            "algorithm": "DP-Bicriteria",
            "approx": "(α,ε)-bicriteria trade-off",
            "privacy": "(ε,δ)-DP",
            "note": "relaxes utility vs privacy split",
        },
    ]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "approximation_table": table_approximation_guarantees(),
        "baselines": ["Non-private MultiGreedy", "Single-objective DP-Greedy × d"],
    }


def evaluation_demo(*, seed: int = 0, k: int = 8) -> dict[str, Any]:
    objectives = toy_three_objective_instance(seed)
    mg = dp_multi_greedy(objectives, k, epsilon=1.0, rng=__import__("numpy").random.default_rng(seed))
    bi = dp_bicriteria_smoke(objectives, k, epsilon=1.0, alpha=0.6, rng=__import__("numpy").random.default_rng(seed + 1))
    return {
        "paper": MosmDpConfig().paper_arxiv,
        "dp_multi_greedy": mg,
        "dp_bicriteria": {"F": bi["F"], "alpha": bi["bicriteria_alpha"]},
        "d": len(objectives),
        "k": k,
    }
