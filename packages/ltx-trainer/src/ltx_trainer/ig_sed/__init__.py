"""IG temporal sound event detection from clip-level classifiers (arXiv:2605.23293)."""

from ltx_trainer.ig_sed.config import DOMESTIC_CLASSES, IgSedConfig
from ltx_trainer.ig_sed.ig import aggregate_to_frames, binarize_percentile, integrated_gradients_1d
from ltx_trainer.ig_sed.layout import LIMITATIONS
from ltx_trainer.ig_sed.metrics import (
    best_percentile,
    frame_f1,
    pointing_game,
    sweep_percentile_metrics,
    temporal_iou,
)
from ltx_trainer.ig_sed.mock import evaluation_smoke
from ltx_trainer.ig_sed.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    figure_3_threshold_sensitivity,
    framework_card,
    headline_results,
    table_1_classification,
    table_2_temporal_detection,
    table_3_per_class,
)

__all__ = [
    "DOMESTIC_CLASSES",
    "IgSedConfig",
    "LIMITATIONS",
    "aggregate_to_frames",
    "benchmarks_bundle",
    "best_percentile",
    "binarize_percentile",
    "evaluation_demo",
    "evaluation_smoke",
    "figure_3_threshold_sensitivity",
    "framework_card",
    "frame_f1",
    "headline_results",
    "integrated_gradients_1d",
    "pointing_game",
    "sweep_percentile_metrics",
    "table_1_classification",
    "table_2_temporal_detection",
    "table_3_per_class",
    "temporal_iou",
]
