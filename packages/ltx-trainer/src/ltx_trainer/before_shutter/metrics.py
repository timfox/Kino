"""Actionability + Bradley–Terry preference metrics (Sec. 4.3, Eq. 6–10)."""

from __future__ import annotations

import math
from typing import Iterable

from ltx_trainer.before_shutter.config import BeforeShutterConfig, PortraitPlanState
from ltx_trainer.before_shutter.human import batch_actionability


def bradley_terry_win_probability(beta_i: float, beta_j: float) -> float:
    """Pr(i ≻ j) — paper Eq. (10)."""
    ei = math.exp(beta_i)
    ej = math.exp(beta_j)
    return ei / (ei + ej)


def fit_bradley_terry_from_wins(
    methods: list[str],
    win_counts: dict[tuple[str, str], int],
) -> dict[str, float]:
    """Simple BT point estimates from pairwise win counts."""
    scores = {m: 0.0 for m in methods}
    for (a, b), wins in win_counts.items():
        if a not in scores or b not in scores:
            continue
        total = max(wins + win_counts.get((b, a), 0), 1)
        scores[a] += math.log((wins + 0.5) / total)
        scores[b] += math.log(((total - wins) + 0.5) / total)
    mean = sum(scores.values()) / max(len(scores), 1)
    return {k: round(v - mean, 2) for k, v in scores.items()}


def table1_main_results() -> list[dict[str, str | float]]:
    """Paper Table 1 — point estimates (human + MLLM overall in last cols)."""
    return [
        {
            "method": "Random Planner",
            "R_coll": 1.00,
            "R_bal": 0.00,
            "V_exp": 1.08,
            "overall_mllm": -2.13,
            "overall_human": -1.43,
        },
        {
            "method": "Template Photographer",
            "R_coll": 0.95,
            "R_bal": 0.02,
            "V_exp": 0.93,
            "overall_mllm": -1.71,
            "overall_human": -0.72,
        },
        {
            "method": "Image-Only Planner",
            "R_coll": 0.84,
            "R_bal": 0.18,
            "V_exp": 1.32,
            "overall_mllm": 0.12,
            "overall_human": 0.15,
        },
        {
            "method": "Spatial-Graph Planner",
            "R_coll": 0.94,
            "R_bal": 0.58,
            "V_exp": 1.49,
            "overall_mllm": 0.84,
            "overall_human": 0.47,
        },
        {
            "method": "Photographic-Graph One-Pass",
            "R_coll": 0.82,
            "R_bal": 0.42,
            "V_exp": 1.41,
            "overall_mllm": 0.50,
            "overall_human": 0.01,
        },
        {
            "method": "Photographic-Graph Greedy",
            "R_coll": 0.80,
            "R_bal": 0.52,
            "V_exp": 1.35,
            "overall_mllm": 1.07,
            "overall_human": 0.54,
        },
        {
            "method": "Ours Full",
            "R_coll": 0.92,
            "R_bal": 0.56,
            "V_exp": 1.56,
            "overall_mllm": 1.30,
            "overall_human": 0.96,
        },
    ]


def table2_stage_ablation() -> list[dict[str, str | float]]:
    """Paper Table 2 — stage-wise ablation (MLLM β)."""
    return [
        {"method": "no-op", "staging": float("nan"), "composition": -0.39, "lighting": -0.31},
        {"method": "+F (frontier only)", "staging": -0.25, "composition": -0.02, "lighting": -0.27},
        {"method": "+G (graph only)", "staging": -0.05, "composition": -0.14, "lighting": 0.12},
        {"method": "+(F, G)", "staging": 0.29, "composition": 0.55, "lighting": 0.46},
    ]


def evaluate_plan_batch(
    states: Iterable[PortraitPlanState],
    *,
    cfg: BeforeShutterConfig | None = None,
) -> dict[str, float]:
    lst = list(states)
    out = batch_actionability(lst, cfg=cfg)
    return out
