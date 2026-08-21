"""DSTA: decoupling spatio-temporal adapter for fine-grained badminton TAL (arXiv:2605.23355)."""

from ltx_trainer.dsta.adapter import dsta_forward_scalar
from ltx_trainer.dsta.config import DSTAConfig, FineBadmintonStats
from ltx_trainer.dsta.dst import dst_forward, split_channels
from ltx_trainer.dsta.layout import LIMITATIONS
from ltx_trainer.dsta.pipeline import (
    evaluation_demo,
    framework_card,
    pipeline_demo,
    table_ablation_branches,
    table_alpha_sensitivity,
    table_dataset_comparison,
    table_efficiency,
    table_map_main,
)
from ltx_trainer.dsta.tal import shuttleset_stroke_interval, temporal_iou

__all__ = [
    "DSTAConfig",
    "FineBadmintonStats",
    "LIMITATIONS",
    "dst_forward",
    "dsta_forward_scalar",
    "evaluation_demo",
    "framework_card",
    "pipeline_demo",
    "shuttleset_stroke_interval",
    "split_channels",
    "table_ablation_branches",
    "table_alpha_sensitivity",
    "table_dataset_comparison",
    "table_efficiency",
    "table_map_main",
    "temporal_iou",
]
