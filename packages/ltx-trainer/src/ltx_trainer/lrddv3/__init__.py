"""LRDDv3 long-range drone detection dataset (Peterson et al., arXiv:2605.25942)."""

from ltx_trainer.lrddv3.config import LRDDv3Config
from ltx_trainer.lrddv3.pipeline import (
    evaluation_demo,
    framework_card,
    split_summary,
    table_dataset_comparison,
    table_detfly_yolov11_benchmark,
    table_resolution_benchmark,
    training_step_demo,
)
from ltx_trainer.lrddv3.range import (
    batch_drone_range_m,
    drone_range_m,
    haversine_horizontal_m,
    vertical_distance_m,
)

__all__ = [
    "LRDDv3Config",
    "batch_drone_range_m",
    "drone_range_m",
    "evaluation_demo",
    "framework_card",
    "haversine_horizontal_m",
    "split_summary",
    "table_dataset_comparison",
    "table_detfly_yolov11_benchmark",
    "table_resolution_benchmark",
    "training_step_demo",
    "vertical_distance_m",
]
