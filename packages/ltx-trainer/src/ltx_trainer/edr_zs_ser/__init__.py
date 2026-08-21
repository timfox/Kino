"""Emotion-discriminative zero-shot cross-lingual SER (arXiv:2606.06200)."""

from ltx_trainer.edr_zs_ser.config import EdrZsSerConfig
from ltx_trainer.edr_zs_ser.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.edr_zs_ser.fold import annotate_audio_save_data
from ltx_trainer.edr_zs_ser.losses import (
    emotion_ce_loss,
    language_aware_weight,
    speaker_adversarial_loss,
    supervised_contrastive_loss,
    total_loss,
)
from ltx_trainer.edr_zs_ser.mock import evaluation_smoke
from ltx_trainer.edr_zs_ser.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table1_crosslingual_settings,
    table2_results,
)

__all__ = [
    "EdrZsSerConfig",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "emotion_ce_loss",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "language_aware_weight",
    "pipeline_demo",
    "pipeline_demo_export",
    "speaker_adversarial_loss",
    "supervised_contrastive_loss",
    "table1_crosslingual_settings",
    "table2_results",
    "total_loss",
]
