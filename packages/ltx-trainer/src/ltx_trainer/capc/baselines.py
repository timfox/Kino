"""CAPC paper anchors (LongBench-v2 + τ-bench + crossover) — arXiv:2607.15516."""

from __future__ import annotations

from typing import Any

# Table 7 condensed: CAPC cheapest in 16/16 (doc × ratio) cells.
LONGBENCH_DOMINANCE = {
    "tasks": 16,
    "configurations": 16,
    "capc_wins_or_ties": 16,
    "dominance_frac": 1.0,
    "mean_cost_usd": 0.0073,
    "mean_savings_vs_cache_only": 0.485,
    "mean_savings_vs_query_aware": 0.644,
    "mean_savings_vs_vanilla": 0.896,
}

# Mean row from Table 7 (USD per query).
LONGBENCH_MEAN_ROW = {
    "A_vanilla": 0.0741,
    "B_cache_only": 0.0144,
    "C_query_aware": 0.0238,
    "D_capc": 0.0073,
}

# Table 3: Sonnet 4.6 ρ_cross(r) crossover thresholds.
RHO_CROSS_TABLE = {
    2: 0.652,
    3: 0.797,
    4: 0.870,
    6: 0.942,
    8: 0.978,
    10: 1.000,
}

# Table 14: τ-bench retail (50 tasks, deterministic DB-state reward).
TAU_BENCH_RETAIL = {
    "n_tasks": 50,
    "vanilla_reward": 36,
    "capc_reward": 36,
    "cache_only_reward": 37,
    "query_aware_reward": 38,
    "vanilla_avg_usd": 0.1244,
    "cache_only_avg_usd": 0.1252,
    "query_aware_avg_usd": 0.1744,
    "capc_avg_usd": 0.1145,
    "capc_vs_vanilla_frac": -0.079,
    "query_aware_vs_vanilla_frac": 0.401,
    "capc_success_delta": 0.0,
    "capc_cost_reduction_frac": 0.079,
    "query_aware_cost_reduction_frac": -0.401,
    "note": "Paper anchors; not a live τ-bench run. Reward CAPC=vanilla=36/50.",
}

PAPER_ANCHORS: dict[str, Any] = {
    "hot_tier_tokens": 3500,
    "rho_hot": 0.83,
    "longbench_dominance": 16,
    "longbench_tasks": 16,
    "longbench_mean_savings_vs_cache_only": 0.485,
    "longbench_mean_savings_vs_query_aware": 0.644,
    "longbench_mean_savings_vs_vanilla": 0.896,
    "tau_cost_reduction_frac": 0.079,
    "tau_query_aware_penalty_frac": 0.401,
    "tau_reward_capc": 36,
    "tau_reward_vanilla": 36,
    "rho_cross_r6": 0.942,
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": "arXiv:2607.15516",
        "longbench": LONGBENCH_DOMINANCE,
        "longbench_mean": LONGBENCH_MEAN_ROW,
        "rho_cross_table": RHO_CROSS_TABLE,
        "tau_bench_retail": TAU_BENCH_RETAIL,
        "anchors": PAPER_ANCHORS,
    }
