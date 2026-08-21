"""SpeechJBB LALM code-switched audio jailbreak benchmark (arXiv:2606.06037)."""

from ltx_trainer.speechjbb.config import SpeechJbbConfig
from ltx_trainer.speechjbb.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.speechjbb.fold import annotate_audio_save_data
from ltx_trainer.speechjbb.metrics import classify_judge_label, rates_from_counts
from ltx_trainer.speechjbb.mock import evaluation_smoke
from ltx_trainer.speechjbb.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table1_synthesis_quality,
    table2_cs_utmos,
    table3_model_results,
    table4_pseudo_word_obfuscation,
    table6_mgsm,
)

__all__ = [
    "SpeechJbbConfig",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "classify_judge_label",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "pipeline_demo",
    "pipeline_demo_export",
    "rates_from_counts",
    "table1_synthesis_quality",
    "table2_cs_utmos",
    "table3_model_results",
    "table4_pseudo_word_obfuscation",
    "table6_mgsm",
]
