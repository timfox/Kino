"""End-to-end VISA pipeline demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.visa.config import VisaConfig
from ltx_trainer.visa.metrics import aggregate_accuracy, mc_accuracy, rubrics_components
from ltx_trainer.visa.pipeline import evaluation_demo
from ltx_trainer.visa.routing import disagree_then_route
from ltx_trainer.visa.voting import ensemble_vote_inference


def pipeline_demo(*, seed: int = 0, cfg: VisaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VisaConfig()
    demo = evaluation_demo(seed=seed, cfg=cfg)
    cases = [
        ("duration_estimation", "C", "B", "C", "C"),
        ("speech_content", "A", "B", "A", "A"),
        ("rhythm_beat", "B", "D", "B", "B"),
    ]
    acc_scores: list[float] = []
    routes = []
    for i, (cat, gold, q, s, vlm) in enumerate(cases):
        vote = ensemble_vote_inference(choices=["A", "B", "C", "D"], seed=seed + i, cfg=cfg)
        route = disagree_then_route(
            category_id=cat,
            qwen_answer=q,
            step_answer=s,
            vlm_answer=vlm,
            llm_selected=q,
        )
        acc_scores.append(mc_accuracy(route["final_answer"], gold))
        routes.append(route)

    rubrics = rubrics_components(factuality=0.88, coherence=0.72, completeness=0.69)
    return {
        "num_stages": 3,
        "demo": demo,
        "routes": routes,
        "accuracy_pct": aggregate_accuracy(acc_scores),
        "rubrics": rubrics,
        "visa_anchor_accuracy": cfg.accuracy,
    }


def eval_smoke() -> dict[str, Any]:
    out = pipeline_demo(seed=42)
    assert out["accuracy_pct"] >= 66.0
    assert out["rubrics"]["rubrics_score"] > 60.0
    return {"status": "ok", "accuracy_pct": out["accuracy_pct"], "rubrics": out["rubrics"]["rubrics_score"]}
