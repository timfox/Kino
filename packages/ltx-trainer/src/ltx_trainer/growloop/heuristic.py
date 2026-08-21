"""Heuristic Learning for rubric optimization (GrowLoop Algorithm 1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ltx_trainer.growloop.config import GrowLoopConfig, RubricTrack


@dataclass
class HeuristicState:
    rubric_version: int
    agreement_rate: float
    iteration: int
    track: RubricTrack


def heuristic_learning_step(
    agreement: float,
    target: float,
    *,
    iteration: int,
    track: RubricTrack,
) -> HeuristicState:
    """Single Compare→Diagnose→Update step proxy."""
    gap = max(0.0, target - agreement)
    step = min(0.06, max(0.015, gap * 0.45))
    improved = agreement + step
    return HeuristicState(
        rubric_version=iteration + 1,
        agreement_rate=min(improved, 1.0),
        iteration=iteration + 1,
        track=track,
    )


def run_heuristic_learning(
    initial: float,
    target: float,
    *,
    max_iters: int = 12,
    track: RubricTrack = RubricTrack.QUALITY,
) -> list[HeuristicState]:
    """Iterate until agreement ≥ target or max_iters."""
    history: list[HeuristicState] = []
    rate = initial
    for i in range(max_iters):
        if rate >= target:
            break
        state = heuristic_learning_step(rate, target, iteration=i, track=track)
        history.append(state)
        rate = state.agreement_rate
    return history


def cascaded_final_score(safety_fatal: bool, quality_raw: float) -> int:
    """Cascaded integration: 0 fatal → else quality bucket {0,1,2}."""
    if safety_fatal:
        return 0
    if quality_raw >= 3.9:
        return 2
    return 1


def stage_pipeline(cfg: GrowLoopConfig | None = None) -> list[dict[str, Any]]:
    cfg = cfg or GrowLoopConfig()
    return [
        {
            "stage": "Rubric cold-start",
            "module": "Rubric Generation Phase 1",
            "outputs": ["Rubric_safety", "Rubric_quality"],
            "seed_n": cfg.seed_cases,
        },
        {
            "stage": "Rubric Heuristic Learning",
            "module": "Algorithm 1",
            "targets": {
                "safety": cfg.safety_convergence_target,
                "quality": cfg.quality_convergence_target,
            },
        },
        {
            "stage": "Case CSP + multi-agent",
            "module": "Case Generation Phases 1–2",
            "csp_fields": cfg.csp_fields,
            "case_target": cfg.case_count,
        },
        {
            "stage": "Dual-loop co-evolution",
            "module": "R↔I triggers",
            "human_seed_only_on_structural_gap": True,
        },
    ]


def heuristic_smoke(cfg: GrowLoopConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GrowLoopConfig()
    safety_hist = run_heuristic_learning(
        0.86,
        cfg.safety_convergence_target,
        max_iters=cfg.safety_iterations + 4,
        track=RubricTrack.SAFETY,
    )
    quality_hist = run_heuristic_learning(
        cfg.quality_initial_agreement,
        cfg.quality_convergence_target,
        max_iters=cfg.quality_iterations + 2,
        track=RubricTrack.QUALITY,
    )
    safety_final = safety_hist[-1].agreement_rate if safety_hist else cfg.safety_final_agreement
    quality_final = quality_hist[-1].agreement_rate if quality_hist else cfg.quality_final_agreement
    return {
        "safety_iterations": len(safety_hist),
        "quality_iterations": len(quality_hist),
        "safety_final_agreement": safety_final,
        "quality_final_agreement": quality_final,
        "safety_converged": safety_final >= cfg.safety_convergence_target,
        "quality_converged": quality_final >= cfg.quality_convergence_target,
        "cascaded_excellent": cascaded_final_score(False, 4.1) == 2,
        "cascaded_fatal": cascaded_final_score(True, 4.5) == 0,
    }
