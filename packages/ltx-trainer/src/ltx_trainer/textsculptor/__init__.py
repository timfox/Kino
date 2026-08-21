"""TextSculptor scene text editing (Lin et al., arXiv:2605.21090)."""

from ltx_trainer.textsculptor.config import TASK_TYPES, VQ_CRITERIA, TextSculptorConfig
from ltx_trainer.textsculptor.data import EditingPair, bench_card, dataset_card
from ltx_trainer.textsculptor.metrics import (
    average_score,
    background_preservation_ssim,
    text_accuracy,
    visual_quality_score,
)
from ltx_trainer.textsculptor.pipeline import (
    evaluation_demo,
    framework_card,
    table_ablation,
    table_main_results,
    table_per_task_textsculptor,
)

__all__ = [
    "EditingPair",
    "TASK_TYPES",
    "TextSculptorConfig",
    "VQ_CRITERIA",
    "average_score",
    "background_preservation_ssim",
    "bench_card",
    "dataset_card",
    "evaluation_demo",
    "framework_card",
    "table_ablation",
    "table_main_results",
    "table_per_task_textsculptor",
    "text_accuracy",
    "visual_quality_score",
]
