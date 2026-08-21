"""SARL spatial audio representation probing (Chen et al., arXiv:2606.05544)."""

from ltx_trainer.sarl.config import SarlConfig
from ltx_trainer.sarl.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.sarl.mock import evaluation_smoke
from ltx_trainer.sarl.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    fig2_aggregates,
    fig3_sensitivity,
    framework_card,
    headline_results,
    pipeline_demo,
    table1_encoders,
)
from ltx_trainer.sarl.probing import aggregate_group, baseline_normalize, linear_probe_predict
from ltx_trainer.sarl.sensitivity import batch_sensitivity, sensitivity_delta
from ltx_trainer.sarl.tasks import ALL_TASKS, ENCODERS, ROOM_TASKS, SOURCE_TASKS

__all__ = [
    "ALL_TASKS",
    "ENCODERS",
    "ROOM_TASKS",
    "SOURCE_TASKS",
    "SarlConfig",
    "aggregate_group",
    "annotate_audio_save_data",
    "baseline_normalize",
    "batch_sensitivity",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "fig2_aggregates",
    "fig3_sensitivity",
    "framework_card",
    "headline_results",
    "linear_probe_predict",
    "pipeline_demo",
    "pipeline_demo_export",
    "sensitivity_delta",
    "table1_encoders",
]

from ltx_trainer.sarl.fold import annotate_audio_save_data  # noqa: E402
