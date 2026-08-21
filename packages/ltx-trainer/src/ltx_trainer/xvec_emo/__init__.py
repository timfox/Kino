"""X-vector emotion arithmetic for Qwen3-TTS (Brito et al., arXiv:2606.05367)."""

from ltx_trainer.xvec_emo.arithmetic import apply_tau, cosine, emotion_tau, interpolate, l2_norm
from ltx_trainer.xvec_emo.config import XvecEmoConfig
from ltx_trainer.xvec_emo.elimination import ELIMINATION_STEPS, localized_operand
from ltx_trainer.xvec_emo.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.xvec_emo.metrics import delta_eecs, meets_identity_floor
from ltx_trainer.xvec_emo.mock import evaluation_smoke
from ltx_trainer.xvec_emo.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    pipeline_demo,
    table1_elimination,
    table2_en_held_out,
    table3_ptbr,
)

__all__ = [
    "ELIMINATION_STEPS",
    "XvecEmoConfig",
    "annotate_audio_save_data",
    "apply_tau",
    "benchmarks_bundle",
    "cosine",
    "delta_eecs",
    "emotion_tau",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "interpolate",
    "l2_norm",
    "localized_operand",
    "meets_identity_floor",
    "pipeline_demo",
    "pipeline_demo_export",
    "table1_elimination",
    "table2_en_held_out",
    "table3_ptbr",
]

from ltx_trainer.xvec_emo.fold import annotate_audio_save_data  # noqa: E402
