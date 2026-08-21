"""Execute rubric↔case dual-loop rounds (GrowLoop Table 4)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.growloop.case_store import case_discriminability, load_cases
from ltx_trainer.growloop.config import GrowLoopConfig
from ltx_trainer.growloop.dual_loop import TriggerCell, select_trigger
from ltx_trainer.growloop.heuristic import run_heuristic_learning
from ltx_trainer.growloop.rubric_editor import default_rubric, refine_rubric


def run_dual_loop(
    *,
    rounds: int = 6,
    seed: int = 42,
    cfg: GrowLoopConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or GrowLoopConfig()
    rubric = default_rubric(cfg)
    cases = load_cases()
    history: list[dict[str, Any]] = []
    agreement = cfg.quality_initial_agreement

    for r in range(rounds):
        disc = case_discriminability(cases)
        dim_consistency = min(1.0, agreement + 0.02 * r)
        trigger = select_trigger(
            dim_consistency=dim_consistency,
            novel_failure=(r == 2),
            discriminability=disc,
            model_saturated=(r >= 4 and disc > 0.35),
        )
        action = "idle"
        if trigger is not None:
            baseline = [3.5, 3.5, 3.5]
            updated = refine_rubric(rubric, trigger, baseline_scores=baseline, cfg=cfg)
            if updated is not None:
                rubric = updated
                action = trigger.value
        hist = run_heuristic_learning(agreement, cfg.quality_convergence_target, max_iters=1)
        if hist:
            agreement = hist[-1].agreement_rate
        history.append({"round": r, "trigger": trigger.value if trigger else None, "action": action, "agreement": agreement})

    return {
        "rounds": rounds,
        "final_rubric_version": rubric.version,
        "final_dimensions": len(rubric.dimensions),
        "final_agreement": agreement,
        "history": history,
        "actions_taken": sum(1 for h in history if h["action"] != "idle"),
    }


def dual_loop_runner_smoke(cfg: GrowLoopConfig | None = None) -> dict[str, Any]:
    out = run_dual_loop(rounds=5, cfg=cfg)
    return {
        "actions_taken": out["actions_taken"],
        "final_agreement": out["final_agreement"],
        "rubric_grew": out["final_dimensions"] > (cfg or GrowLoopConfig()).quality_dimensions,
    }
