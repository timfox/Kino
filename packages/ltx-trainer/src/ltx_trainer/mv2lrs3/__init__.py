"""MV2LRS3 — matched AVSR generalisability benchmark (arXiv:2606.07259)."""

from ltx_trainer.mv2lrs3.config import Mv2Lrs3Config
from ltx_trainer.mv2lrs3.eval import eval_smoke, pipeline_demo
from ltx_trainer.mv2lrs3.fold import annotate_audio_save_data
from ltx_trainer.mv2lrs3.ltx_plan import ltx_integration_plan
from ltx_trainer.mv2lrs3.matching import knn_match_one, matching_demo
from ltx_trainer.mv2lrs3.metrics import iwer, modality_delta, predict_mv2lrs3_wer
from ltx_trainer.mv2lrs3.mock import evaluation_smoke
from ltx_trainer.mv2lrs3.models import model_registry
from ltx_trainer.mv2lrs3.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    linear_fit_points,
    table1_lrs3_vs_mv2lrs3,
    table2_ten_x_subset,
    table3_leave_one_out,
    table4_binned_subsets,
    table5_vocabulary_iwer,
    table6_modalities,
    table7_error_rates,
)

__all__ = [
    "Mv2Lrs3Config",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "iwer",
    "knn_match_one",
    "linear_fit_points",
    "ltx_integration_plan",
    "matching_demo",
    "modality_delta",
    "model_registry",
    "pipeline_demo",
    "predict_mv2lrs3_wer",
    "table1_lrs3_vs_mv2lrs3",
    "table2_ten_x_subset",
    "table3_leave_one_out",
    "table4_binned_subsets",
    "table5_vocabulary_iwer",
    "table6_modalities",
    "table7_error_rates",
]
