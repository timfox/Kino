"""PINNS pedestrian–vehicle interaction dataset (Peng et al., arXiv:2605.25947)."""

from ltx_trainer.pinns.config import PINNSConfig
from ltx_trainer.pinns.homography import (
    apply_homography,
    estimate_homography,
    reconstruction_error_m,
    reprojection_error,
)
from ltx_trainer.pinns.metrics import ade, batch_ade_fde, fde
from ltx_trainer.pinns.pipeline import (
    evaluation_demo,
    framework_card,
    table_baseline_trajectron,
    table_calibration_stats,
    table_dataset_comparison,
    training_step_demo,
)

__all__ = [
    "PINNSConfig",
    "ade",
    "apply_homography",
    "batch_ade_fde",
    "evaluation_demo",
    "estimate_homography",
    "fde",
    "framework_card",
    "reconstruction_error_m",
    "reprojection_error",
    "table_baseline_trajectron",
    "table_calibration_stats",
    "table_dataset_comparison",
    "training_step_demo",
]
