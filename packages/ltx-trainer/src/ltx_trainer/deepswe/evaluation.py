"""DeepSWE evaluation demos and smoke metrics."""

from __future__ import annotations

from typing import Any

from ltx_trainer.deepswe.benchmarks import benchmarks_bundle
from ltx_trainer.deepswe.config import DeepSWEConfig
from ltx_trainer.deepswe.core import deepswe_leaderboard, leaderboard_ranked, separation_vs_swe_bench_pro
from ltx_trainer.deepswe.harness import harness_card, harness_pilot_comparison
from ltx_trainer.deepswe.methodology import methodology_card
from ltx_trainer.deepswe.qualitative import qualitative_bundle
from ltx_trainer.deepswe.verifier import example_task_boa_cancellation, verifier_audit_summary


def grade_example_agent_patch(*, then_without_handle: int = 1) -> dict[str, Any]:
    """Simulate blog Boa cancellation failure (orphan Promise.then callbacks)."""
    task = example_task_boa_cancellation()
    state = {
        "cancelled_eval_pending_jobs": 0,
        "then_callbacks_without_handle": then_without_handle,
        "unrelated_jobs_completed": 2,
        "regression_ok": True,
    }
    return task.grade(state)


def evaluation_demo(cfg: DeepSWEConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DeepSWEConfig()
    lb = deepswe_leaderboard()
    top = leaderboard_ranked()[0]
    grade_fail = grade_example_agent_patch(then_without_handle=1)
    grade_ok = grade_example_agent_patch(then_without_handle=0)
    return {
        "leaderboard_top": {"model": top[0], "pass_pct": top[1]},
        "leaderboard_size": len(lb),
        "separation": separation_vs_swe_bench_pro(),
        "verifier_audit": verifier_audit_summary(),
        "behavioral_grade_miss": grade_fail,
        "behavioral_grade_pass": grade_ok,
        "methodology": methodology_card(),
        "harness": harness_card(),
        "harness_pilot": harness_pilot_comparison(),
        "qualitative": qualitative_bundle(),
        "paper_tables": benchmarks_bundle(),
        "github": cfg.github_repo,
        "harness_name": cfg.harness,
    }


def evaluation_smoke(cfg: DeepSWEConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DeepSWEConfig()
    ev = evaluation_demo(cfg)
    assert ev["behavioral_grade_miss"]["passed"] is False
    assert ev["behavioral_grade_pass"]["passed"] is True
    assert ev["leaderboard_size"] >= 10
    return {
        "package": "ltx_trainer.deepswe",
        "status": "smoke_ok",
        "paper": cfg.title,
        "website": cfg.website,
        "ok": True,
        "n_tasks": cfg.n_tasks,
        "top_model_pass_pct": ev["leaderboard_top"]["pass_pct"],
        "deepswe_span_pp": ev["separation"]["deepswe_span_pp"],
    }
