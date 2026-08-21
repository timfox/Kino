"""UniVoice unified speech/singing CFM (arXiv:2606.05852)."""

from ltx_trainer.univoice.cfm import cfm_loss, euler_sample_step, ot_interpolant, target_velocity
from ltx_trainer.univoice.conditioning import (
    FactorizedCondition,
    TaskModality,
    axis_guidance_delta,
    build_condition,
    concat_conditions,
    melody_from_midi,
    null_melody_token,
)
from ltx_trainer.univoice.config import UniVoiceConfig
from ltx_trainer.univoice.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.univoice.mock import evaluation_smoke
from ltx_trainer.univoice.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table1_main_results,
    table2_ablation,
    unisinging_eval_summary,
)

__all__ = [
    "FactorizedCondition",
    "TaskModality",
    "UniVoiceConfig",
    "annotate_audio_save_data",
    "axis_guidance_delta",
    "benchmarks_bundle",
    "build_condition",
    "cfm_loss",
    "concat_conditions",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "euler_sample_step",
    "framework_card",
    "headline_results",
    "melody_from_midi",
    "null_melody_token",
    "ot_interpolant",
    "pipeline_demo",
    "pipeline_demo_export",
    "table1_main_results",
    "table2_ablation",
    "target_velocity",
    "unisinging_eval_summary",
]

from ltx_trainer.univoice.fold import annotate_audio_save_data  # noqa: E402
