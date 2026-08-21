"""Five hard gates for case-set verification (GrowLoop Table 5)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.growloop.config import GrowLoopConfig
from ltx_trainer.growloop.metrics import cliffs_delta, kendall_tau


def gate_registry(cfg: GrowLoopConfig | None = None) -> list[dict[str, Any]]:
    cfg = cfg or GrowLoopConfig()
    return [
        {
            "property": "Diversity",
            "metric": "Diversity Score",
            "threshold": f"≥ {cfg.gate_diversity_min}",
            "realized": cfg.diversity_score,
        },
        {
            "property": "Ranking consistency",
            "metric": "Kendall τ̄",
            "threshold": f"≥ {cfg.gate_kendall_tau_min}",
            "realized": cfg.kendall_tau_mean,
        },
        {
            "property": "Discriminability",
            "metric": "Cliff's δ_min",
            "threshold": f"≥ {cfg.gate_cliffs_delta_min}",
            "realized": cfg.cliffs_delta_min,
        },
        {
            "property": "Discriminability",
            "metric": "Δ_adj",
            "threshold": f"≥ {cfg.gate_adjacent_gap_min}",
            "realized": 11.4,
            "note": "tightest best→good gap",
        },
        {
            "property": "Difficulty calibration",
            "metric": "S̄_best",
            "threshold": f"[{cfg.gate_best_mean_lo}, {cfg.gate_best_mean_hi}]",
            "realized": cfg.best_tier_mean,
        },
    ]


def evaluate_gates(
    *,
    diversity_score: float,
    kendall_tau_mean: float,
    cliffs_delta_min: float,
    adjacent_gap_min: float,
    best_tier_mean: float,
    cfg: GrowLoopConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or GrowLoopConfig()
    checks = {
        "diversity": diversity_score >= cfg.gate_diversity_min,
        "kendall_tau": kendall_tau_mean >= cfg.gate_kendall_tau_min,
        "cliffs_delta": cliffs_delta_min >= cfg.gate_cliffs_delta_min,
        "adjacent_gap": adjacent_gap_min >= cfg.gate_adjacent_gap_min,
        "best_mean": cfg.gate_best_mean_lo <= best_tier_mean <= cfg.gate_best_mean_hi,
    }
    return {"pass": all(checks.values()), "checks": checks}


def gates_smoke(cfg: GrowLoopConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GrowLoopConfig()
    tier_means = list(cfg.tier_means)
    per_case_tau = kendall_tau(tier_means)
    gaps = [tier_means[i] - tier_means[i + 1] for i in range(len(tier_means) - 1)]
    delta_bg = cliffs_delta([tier_means[0]], [tier_means[1]])
    out = evaluate_gates(
        diversity_score=cfg.diversity_score,
        kendall_tau_mean=cfg.kendall_tau_mean,
        cliffs_delta_min=cfg.cliffs_delta_min,
        adjacent_gap_min=min(gaps),
        best_tier_mean=cfg.best_tier_mean,
        cfg=cfg,
    )
    out["per_case_tau_toy"] = per_case_tau
    out["delta_best_good_toy"] = delta_bg
    return out
