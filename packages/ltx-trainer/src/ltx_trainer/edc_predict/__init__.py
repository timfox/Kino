"""Multi-band EDC prediction from room features — stub (arXiv:2605.20968)."""

from ltx_trainer.edc_predict.config import EdcPredictConfig
from ltx_trainer.edc_predict.layout import LIMITATIONS
from ltx_trainer.edc_predict.loss import composite_loss, slope_d_db, to_db
from ltx_trainer.edc_predict.mock import evaluation_smoke
from ltx_trainer.edc_predict.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table1_room_feature_ranges,
    table2_convnet_metrics,
)
from ltx_trainer.edc_predict.rss import random_sign_sticky_sequence

__all__ = [
    "LIMITATIONS",
    "EdcPredictConfig",
    "benchmarks_bundle",
    "composite_loss",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "random_sign_sticky_sequence",
    "slope_d_db",
    "table1_room_feature_ranges",
    "table2_convnet_metrics",
    "to_db",
]
