"""LiveSVG: zero-shot SVG animation via target-video fitting (arXiv:2605.30174)."""

from ltx_trainer.livesvg.challengesvg import ChallengeSVGRecord, challengesvg_manifest, example_records_preview
from ltx_trainer.livesvg.bundle import export_motion_bundle, load_motion_bundle
from ltx_trainer.livesvg.experiments import export_experiment_bundle, run_full_experiment, run_synthetic_fitting_experiment
from ltx_trainer.livesvg.fitting import effective_progressive_interval
from ltx_trainer.livesvg.paper_report import compare_experiment_to_paper, format_report_markdown
from ltx_trainer.livesvg.ltx_bridge import build_ltx_i2v_kwargs, ltx_integration_notes
from ltx_trainer.livesvg.svg_io import count_drawable_paths, load_svg_summary, recolor_svg_paths
from ltx_trainer.livesvg.target_video import clean_target_background, synthetic_translating_disk
from ltx_trainer.livesvg.config import BASELINES, LiveSVGConfig
from ltx_trainer.livesvg.fitting import (
    copy_keyframe_motion,
    pipeline_stage_summary,
    progressive_activation_schedule,
    toy_synthetic_fitting_step,
    tracking_init_fallback,
)
from ltx_trainer.livesvg.homography import (
    apply_homography_points,
    compose_path_motion,
    homography_from_similarity,
    identity_homography,
)
from ltx_trainer.livesvg.losses import (
    blurred_mse_loss,
    foreground_containment_penalty,
    g1_joint_penalty,
    gaussian_blur_nchw,
    livesvg_total_loss,
    spatial_offset_regularizer,
)
from ltx_trainer.livesvg.paper_tables import (
    human_preference_rates,
    table_ablation_gemini,
    table_aniclipart_quantitative,
    table_benchmark_structure,
    table_challengesvg_quantitative,
    table_runtime_minutes,
)
from ltx_trainer.livesvg.pipeline import benchmark_manifest, evaluation_demo, framework_card, paper_limitations
from ltx_trainer.livesvg.prompts import (
    gemini_stage1_rubric,
    gemini_stage2_instruction,
    i2v_motion_prompt,
    method_capability_table,
)
from ltx_trainer.livesvg.recolor import (
    assign_path_palette,
    farthest_point_rgb_palette,
    min_pairwise_rgb_distance,
    recolorization_report,
)

__all__ = [
    "BASELINES",
    "ChallengeSVGRecord",
    "LiveSVGConfig",
    "apply_homography_points",
    "assign_path_palette",
    "benchmark_manifest",
    "blurred_mse_loss",
    "build_ltx_i2v_kwargs",
    "challengesvg_manifest",
    "clean_target_background",
    "compose_path_motion",
    "count_drawable_paths",
    "copy_keyframe_motion",
    "compare_experiment_to_paper",
    "effective_progressive_interval",
    "evaluation_demo",
    "export_experiment_bundle",
    "export_motion_bundle",
    "format_report_markdown",
    "example_records_preview",
    "farthest_point_rgb_palette",
    "framework_card",
    "foreground_containment_penalty",
    "g1_joint_penalty",
    "gaussian_blur_nchw",
    "gemini_stage1_rubric",
    "gemini_stage2_instruction",
    "homography_from_similarity",
    "human_preference_rates",
    "i2v_motion_prompt",
    "identity_homography",
    "load_motion_bundle",
    "load_svg_summary",
    "livesvg_total_loss",
    "ltx_integration_notes",
    "method_capability_table",
    "min_pairwise_rgb_distance",
    "paper_limitations",
    "pipeline_stage_summary",
    "progressive_activation_schedule",
    "recolor_svg_paths",
    "recolorization_report",
    "run_full_experiment",
    "run_synthetic_fitting_experiment",
    "spatial_offset_regularizer",
    "synthetic_translating_disk",
    "table_ablation_gemini",
    "table_aniclipart_quantitative",
    "table_benchmark_structure",
    "table_challengesvg_quantitative",
    "table_runtime_minutes",
    "toy_synthetic_fitting_step",
    "tracking_init_fallback",
]
