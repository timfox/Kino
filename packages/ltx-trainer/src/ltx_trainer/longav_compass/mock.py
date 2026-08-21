"""LongAV-Compass evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.longav_compass.judges import evaluate_case
from ltx_trainer.longav_compass.annotation import example_t2av_performance_ads_l4


def evaluation_smoke() -> dict[str, Any]:
    from ltx_trainer.longav_compass.leaderboard import rank_models_t2av
    from ltx_trainer.longav_compass.pipeline import training_step_demo

    report = evaluate_case(example_t2av_performance_ads_l4(), has_audio=True)
    agg = report["aggregate"]
    top = rank_models_t2av()[0]
    step = training_step_demo()
    return {
        "case_id": report["case_id"],
        "vqa": agg["VQA"],
        "vq": agg["VQ"],
        "n_events_scored": len(report["event_vqa"]),
        "leader_t2av": top["model"],
        "human_visual_pearson": step["human_alignment"]["visual_quality"],
    }
