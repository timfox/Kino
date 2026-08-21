"""MVSE query-adaptive AV person retrieval (arXiv:2606.05931)."""

from ltx_trainer.mvse_amd.config import MvseAmdConfig
from ltx_trainer.mvse_amd.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.mvse_amd.features import modality_detection_features
from ltx_trainer.mvse_amd.fold import annotate_audio_save_data
from ltx_trainer.mvse_amd.mock import evaluation_smoke
from ltx_trainer.mvse_amd.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table2_modality_detection,
    table3_retrieval,
    table4_by_presence,
)
from ltx_trainer.mvse_amd.scoring import fused_score, lambda_for_presence, max_pool_file_score

__all__ = [
    "MvseAmdConfig",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "fused_score",
    "headline_results",
    "lambda_for_presence",
    "max_pool_file_score",
    "modality_detection_features",
    "pipeline_demo",
    "pipeline_demo_export",
    "table2_modality_detection",
    "table3_retrieval",
    "table4_by_presence",
]
