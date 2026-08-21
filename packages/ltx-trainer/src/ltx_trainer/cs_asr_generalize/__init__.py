"""CS-ASR generalization to unseen language pairs (arXiv:2606.05846)."""

from ltx_trainer.cs_asr_generalize.config import CsAsrGeneralizeConfig
from ltx_trainer.cs_asr_generalize.domain_gen import DgMethod, dg_objective, fish_alignment, fishr_variance_penalty
from ltx_trainer.cs_asr_generalize.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.cs_asr_generalize.merging import MergeMethod, dare_merge, merge_models, task_arithmetic, ties_merge
from ltx_trainer.cs_asr_generalize.metrics import average_mer, mer
from ltx_trainer.cs_asr_generalize.mock import evaluation_smoke
from ltx_trainer.cs_asr_generalize.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table1_mer_results,
)

__all__ = [
    "CsAsrGeneralizeConfig",
    "DgMethod",
    "MergeMethod",
    "annotate_audio_save_data",
    "average_mer",
    "benchmarks_bundle",
    "dare_merge",
    "dg_objective",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "fish_alignment",
    "fishr_variance_penalty",
    "framework_card",
    "headline_results",
    "mer",
    "merge_models",
    "pipeline_demo",
    "pipeline_demo_export",
    "table1_mer_results",
    "task_arithmetic",
    "ties_merge",
]

from ltx_trainer.cs_asr_generalize.fold import annotate_audio_save_data  # noqa: E402
