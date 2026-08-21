"""LongAV-Compass: minute-scale T2AV / I2AV / V2AV evaluation (arXiv:2605.26244)."""

from ltx_trainer.longav_compass.catalog import benchmark_catalog_outline
from ltx_trainer.longav_compass.annotation import (
    BenchmarkCase,
    EventAnnotation,
    build_generation_prompt,
    case_to_dict,
    challenging_case_summaries,
    example_i2av_performance_ads_l4,
    example_t2av_performance_ads_l4,
    example_v2av_content_creator_l4,
)
from ltx_trainer.longav_compass.config import (
    ApplicationScenario,
    ComplexityLevel,
    LongAVCompassConfig,
    X2AVTask,
)
from ltx_trainer.longav_compass.human_alignment import (
    human_alignment_validation,
    pilot_model_win_rates,
    table_human_alignment_pearson,
)
from ltx_trainer.longav_compass.batch import (
    evaluate_case_object,
    evaluate_checkout_batch,
    evaluate_upstream_case,
)
from ltx_trainer.longav_compass.io import (
    case_from_dict,
    discover_case_ids,
    load_case_from_upstream,
    load_case_json,
    summarize_checkout,
)
from ltx_trainer.longav_compass.judges import evaluate_case, evaluate_examples, judge_event_vqa
from ltx_trainer.longav_compass.scoring import model_win_ratio, pairwise_outcome, pearson_r
from ltx_trainer.longav_compass.layout import LIMITATIONS
from ltx_trainer.longav_compass.leaderboard import (
    diagnostic_gaps_for_model,
    leaderboard_bundle,
    rank_models_i2av,
    rank_models_t2av,
    rank_models_v2av,
)
from ltx_trainer.longav_compass.ltx_bridge import ltx_minute_av_eval_plan, plan_for_case
from ltx_trainer.longav_compass.segmentation import (
    boundary_clip_windows,
    canonical_events_json,
    event_segment_specs,
    ffmpeg_segment_commands,
    parse_time_range,
)
from ltx_trainer.longav_compass.metrics import (
    balanced_score,
    clip_image_alignment,
    clip_text_video_alignment,
    list_audio_metric_definitions,
    list_video_metric_definitions,
    score_event_vqa,
)
from ltx_trainer.longav_compass.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    training_step_demo,
)
from ltx_trainer.longav_compass.tables import (
    table_difficulty_balanced_scores,
    table_event_count_balanced_scores,
    table_i2av_main_results,
    table_input_format_sensitivity,
    table_scenario_balanced_scores_t2av,
    table_t2av_main_results,
    table_v2av_main_results,
)
from ltx_trainer.longav_compass.taxonomy import (
    benchmark_comparison_table,
    scenario_complexity_notes,
    task_coverage_table,
)

__all__ = [
    "ApplicationScenario",
    "BenchmarkCase",
    "ComplexityLevel",
    "EventAnnotation",
    "LIMITATIONS",
    "LongAVCompassConfig",
    "X2AVTask",
    "balanced_score",
    "benchmark_catalog_outline",
    "benchmark_comparison_table",
    "benchmarks_bundle",
    "build_generation_prompt",
    "case_from_dict",
    "case_to_dict",
    "challenging_case_summaries",
    "discover_case_ids",
    "clip_image_alignment",
    "clip_text_video_alignment",
    "evaluate_case",
    "evaluate_case_object",
    "evaluate_checkout_batch",
    "evaluate_examples",
    "evaluate_upstream_case",
    "evaluation_demo",
    "example_i2av_performance_ads_l4",
    "example_t2av_performance_ads_l4",
    "example_v2av_content_creator_l4",
    "framework_card",
    "human_alignment_validation",
    "diagnostic_gaps_for_model",
    "judge_event_vqa",
    "leaderboard_bundle",
    "load_case_from_upstream",
    "load_case_json",
    "model_win_ratio",
    "pairwise_outcome",
    "pearson_r",
    "pilot_model_win_rates",
    "boundary_clip_windows",
    "canonical_events_json",
    "event_segment_specs",
    "ffmpeg_segment_commands",
    "list_audio_metric_definitions",
    "list_video_metric_definitions",
    "ltx_minute_av_eval_plan",
    "parse_time_range",
    "plan_for_case",
    "scenario_complexity_notes",
    "summarize_checkout",
    "score_event_vqa",
    "table_difficulty_balanced_scores",
    "table_event_count_balanced_scores",
    "table_human_alignment_pearson",
    "table_i2av_main_results",
    "table_scenario_balanced_scores_t2av",
    "table_input_format_sensitivity",
    "table_t2av_main_results",
    "table_v2av_main_results",
    "task_coverage_table",
    "training_step_demo",
]
