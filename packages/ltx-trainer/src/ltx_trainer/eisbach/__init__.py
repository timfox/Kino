"""Eisbach log-barrier — DiT belief-space entropy prior (Li & Li, 2026)."""

from ltx_trainer.eisbach.analysis import analysis_demo, structural_metrics
from ltx_trainer.eisbach.barrier import barrier_demo, compute_barrier, log_barrier_weights, scale_loss
from ltx_trainer.eisbach.config import EisbachConfig
from ltx_trainer.eisbach.dora import dora_demo
from ltx_trainer.eisbach.eval import eval_smoke, pipeline_demo
from ltx_trainer.eisbach.fold import annotate_audio_save_data
from ltx_trainer.eisbach.mock import evaluation_smoke
from ltx_trainer.eisbach.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table1_curation_comparison,
    table2_structural_dimensions,
    table_training_config,
    testable_predictions,
)

__all__ = [
    "EisbachConfig",
    "annotate_audio_save_data",
    "analysis_demo",
    "barrier_demo",
    "benchmarks_bundle",
    "compute_barrier",
    "dora_demo",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "log_barrier_weights",
    "pipeline_demo",
    "scale_loss",
    "structural_metrics",
    "table1_curation_comparison",
    "table2_structural_dimensions",
    "table_training_config",
    "testable_predictions",
]
