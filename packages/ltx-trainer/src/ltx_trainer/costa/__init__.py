"""CoSTA — CS-Cond TTS AD augmentation (arXiv:2606.06170)."""

from ltx_trainer.costa.augmentation import (
    augmentation_factor_curve,
    build_text_prompt,
    cosyvoice_prompt,
    f5_cognition_label,
    optimal_augmentation_factor,
    synthesize_stub,
    tta_probability_average,
)
from ltx_trainer.costa.config import CostaConfig
from ltx_trainer.costa.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.costa.fold import annotate_audio_save_data
from ltx_trainer.costa.mock import evaluation_smoke
from ltx_trainer.costa.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table1_tts_objective,
    table2_beat_ratios,
    table2_headline_configs,
    table3_tta,
    table4_comparisons,
)

__all__ = [
    "CostaConfig",
    "annotate_audio_save_data",
    "augmentation_factor_curve",
    "benchmarks_bundle",
    "build_text_prompt",
    "cosyvoice_prompt",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "f5_cognition_label",
    "framework_card",
    "headline_results",
    "optimal_augmentation_factor",
    "pipeline_demo",
    "pipeline_demo_export",
    "synthesize_stub",
    "table1_tts_objective",
    "table2_beat_ratios",
    "table2_headline_configs",
    "table3_tta",
    "table4_comparisons",
    "tta_probability_average",
]
