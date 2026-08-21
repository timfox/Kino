"""Sagnac-assisted enhanced ϕ-OTDR DAS benchmark (arXiv:2606.05754)."""

from ltx_trainer.sagnac_phi_otdr.alignment import cross_correlation, optimal_lag
from ltx_trainer.sagnac_phi_otdr.config import SagnacPhiOtdrConfig
from ltx_trainer.sagnac_phi_otdr.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.sagnac_phi_otdr.grouping import BEST_GROUPING, DEFAULT_CONTIGUOUS, ChannelGrouping
from ltx_trainer.sagnac_phi_otdr.metrics import accuracy, confusion_matrix, macro_f1, nar_fnr
from ltx_trainer.sagnac_phi_otdr.mock import evaluation_smoke
from ltx_trainer.sagnac_phi_otdr.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table2_benchmark,
    table4_grouping_search,
)

__all__ = [
    "BEST_GROUPING",
    "DEFAULT_CONTIGUOUS",
    "ChannelGrouping",
    "SagnacPhiOtdrConfig",
    "accuracy",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "confusion_matrix",
    "cross_correlation",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "macro_f1",
    "nar_fnr",
    "optimal_lag",
    "pipeline_demo",
    "pipeline_demo_export",
    "table2_benchmark",
    "table4_grouping_search",
]

from ltx_trainer.sagnac_phi_otdr.fold import annotate_audio_save_data  # noqa: E402
