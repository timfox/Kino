"""LongAV-Compass framework card, paper tables, evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.longav_compass.annotation import (
    build_generation_prompt,
    case_to_dict,
    challenging_case_summaries,
    example_i2av_performance_ads_l4,
    example_t2av_performance_ads_l4,
    example_v2av_content_creator_l4,
)
from ltx_trainer.longav_compass.config import LongAVCompassConfig
from ltx_trainer.longav_compass.human_alignment import human_alignment_validation, table_human_alignment_pearson
from ltx_trainer.longav_compass.judges import evaluate_case
from ltx_trainer.longav_compass.layout import LIMITATIONS
from ltx_trainer.longav_compass.ltx_bridge import ltx_minute_av_eval_plan, plan_for_case
from ltx_trainer.longav_compass.metrics import balanced_score, list_audio_metric_definitions, list_video_metric_definitions
from ltx_trainer.longav_compass.segmentation import canonical_events_json
from ltx_trainer.longav_compass.tables import (
    evaluated_models_list,
    table_difficulty_balanced_scores,
    table_event_count_balanced_scores,
    table_i2av_main_results,
    table_input_format_sensitivity,
    table_scenario_balanced_scores_t2av,
    table_t2av_main_results,
    table_v2av_main_results,
)
from ltx_trainer.longav_compass.taxonomy import benchmark_comparison_table, scenario_complexity_notes, task_coverage_table


def framework_card(cfg: LongAVCompassConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LongAVCompassConfig()
    return {
        "name": "LongAV-Compass",
        "paper": cfg.paper_arxiv,
        "github": cfg.github,
        "idea": (
            "First unified minute-scale audio-visual benchmark across T2AV, I2AV, and V2AV with "
            "taxonomy-guided cases, event-level annotations, and 20+ diagnostic dimensions "
            "(MLLM judge + DINO-v2 / CLIP / ArcFace / ImageBind complements)."
        ),
        "samples": {
            "total": cfg.n_samples,
            "T2AV": cfg.n_t2av,
            "I2AV": cfg.n_i2av,
            "V2AV": cfg.n_v2av,
        },
        "duration_s": [cfg.target_duration_s_min, cfg.target_duration_s_max],
        "judge": cfg.judge_model,
        "evaluated_models": cfg.n_evaluated_models,
        "perspectives": [
            "within-segment quality",
            "cross-segment consistency",
            "global narrative coherence",
            "audio-visual synchronization",
            "input-conditioned alignment",
        ],
        "human_alignment_pearson": {
            "content_fidelity": cfg.human_alignment_pearson[0],
            "visual_quality": cfg.human_alignment_pearson[1],
            "long_video_stability": cfg.human_alignment_pearson[2],
        },
    }


def benchmarks_bundle() -> dict[str, Any]:
    from ltx_trainer.longav_compass.leaderboard import leaderboard_bundle

    return {
        "table_1_comparison": benchmark_comparison_table(),
        "table_2_tasks": task_coverage_table(),
        "table_3_t2av": table_t2av_main_results(),
        "table_4_i2av": table_i2av_main_results(),
        "table_5_v2av": table_v2av_main_results(),
        "table_6_difficulty": table_difficulty_balanced_scores(),
        "table_7_input_format": table_input_format_sensitivity(),
        "fig_4_scenario_t2av": table_scenario_balanced_scores_t2av(),
        "fig_9_event_count": table_event_count_balanced_scores(),
        "fig_10_human_alignment": table_human_alignment_pearson(),
        "human_alignment_validation": human_alignment_validation(),
        "challenging_cases": challenging_case_summaries(),
        "taxonomy_notes": scenario_complexity_notes(),
        "video_metrics": list_video_metric_definitions(),
        "audio_metrics": list_audio_metric_definitions(),
        "models": evaluated_models_list(),
        "leaderboards": leaderboard_bundle(),
    }


def evaluation_demo(cfg: LongAVCompassConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LongAVCompassConfig()
    examples = [
        example_t2av_performance_ads_l4(),
        example_i2av_performance_ads_l4(),
        example_v2av_content_creator_l4(),
    ]
    reports: list[dict[str, Any]] = []
    for case in examples:
        report = evaluate_case(case, has_audio=True, quality_bias=0.90)
        reports.append(report)

    primary = reports[0]
    agg = primary["aggregate"]
    bal = balanced_score(
        {
            "VQA": agg["VQA"],
            "VQ": agg["VQ"],
            "Cont.": agg["Cont."],
            "Hol.": agg["Hol."],
            "TVAlign": agg["TVAlign"],
            "AVS": agg.get("AVS", 3.0),
        }
    )
    return {
        "framework": framework_card(cfg),
        "example_cases": [case_to_dict(c) for c in examples],
        "generation_prompts": {c.task: build_generation_prompt(c) for c in examples},
        "evaluation_reports": reports,
        "evaluation_report": primary,
        "balanced_score_demo": bal,
        "limitations": list(LIMITATIONS),
        "paper_tables": benchmarks_bundle(),
        "ltx_plan": ltx_minute_av_eval_plan(),
        "canonical_events_example": canonical_events_json(example_t2av_performance_ads_l4()),
        "per_case_plans": {c.task: plan_for_case(c) for c in examples},
    }


def training_step_demo(cfg: LongAVCompassConfig | None = None) -> dict[str, Any]:
    """Reference hook for tooling parity with other trainer stubs (no GPU training)."""
    cfg = cfg or LongAVCompassConfig()
    from ltx_trainer.longav_compass.leaderboard import leaderboard_bundle, rank_models_t2av

    t2av_top = rank_models_t2av()[0]
    return {
        "paper": cfg.paper_arxiv,
        "benchmark_samples": cfg.n_samples,
        "leader_t2av": t2av_top["model"],
        "leader_balanced_score": t2av_top["balanced_score"],
        "human_alignment": table_human_alignment_pearson(cfg),
        "leaderboards": leaderboard_bundle(),
    }
