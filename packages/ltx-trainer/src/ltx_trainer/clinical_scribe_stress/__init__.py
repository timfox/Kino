"""Paired acoustic stress test for ambient clinical scribes (arXiv:2606.05909)."""

from ltx_trainer.clinical_scribe_stress.config import ClinicalScribeStressConfig
from ltx_trainer.clinical_scribe_stress.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.clinical_scribe_stress.metrics import (
    err_prop,
    ins_rate,
    neg_err_rate,
    scer_rate,
    triage_match,
    under_triage_rate,
    unsafe_rate,
    wer,
)
from ltx_trainer.clinical_scribe_stress.mock import evaluation_smoke
from ltx_trainer.clinical_scribe_stress.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table2_main_results,
)

__all__ = [
    "ClinicalScribeStressConfig",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "err_prop",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "ins_rate",
    "neg_err_rate",
    "pipeline_demo",
    "pipeline_demo_export",
    "scer_rate",
    "table2_main_results",
    "triage_match",
    "under_triage_rate",
    "unsafe_rate",
    "wer",
]

from ltx_trainer.clinical_scribe_stress.fold import annotate_audio_save_data  # noqa: E402
