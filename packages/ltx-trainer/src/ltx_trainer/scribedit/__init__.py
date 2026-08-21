"""ScribEdit: scribble-guided image editing utilities (Xu et al., arXiv:2605.25568).

Reference implementation for Coverage-then-Realism curriculum, Multi-Task Mosaicking,
Edit-Focused Loss, and VIBE metric helpers. Qwen-Image-Edit fine-tuning is external.
"""

from ltx_trainer.scribedit.config import ScribEditConfig
from ltx_trainer.scribedit.curriculum import TrainingStage, sample_is_mosaic, stage_training_flags
from ltx_trainer.scribedit.distractor import add_distractor_scribbles, draw_scribble_box
from ltx_trainer.scribedit.losses import (
    edit_focused_fm_loss,
    scribedit_training_loss,
    velocity_target,
    whole_image_fm_loss,
)
from ltx_trainer.scribedit.mask import edit_region_mask, resize_mask_to_latent
from ltx_trainer.scribedit.metrics import (
    baseline_cross_task_score,
    cross_task_transfer_matrix,
    distractor_ablation_delta,
    geometric_mean_score,
    instruction_vs_domain_gap,
    single_task_average,
)
from ltx_trainer.scribedit.mosaic import MosaicSample, format_multi_instruction, multi_task_mosaic
from ltx_trainer.scribedit.pipeline import (
    build_mosaic_training_sample,
    prepare_latent_masks,
    scribedit_fm_step,
)

__all__ = [
    "MosaicSample",
    "ScribEditConfig",
    "TrainingStage",
    "add_distractor_scribbles",
    "baseline_cross_task_score",
    "build_mosaic_training_sample",
    "cross_task_transfer_matrix",
    "distractor_ablation_delta",
    "draw_scribble_box",
    "edit_focused_fm_loss",
    "edit_region_mask",
    "format_multi_instruction",
    "geometric_mean_score",
    "instruction_vs_domain_gap",
    "multi_task_mosaic",
    "prepare_latent_masks",
    "resize_mask_to_latent",
    "sample_is_mosaic",
    "scribedit_fm_step",
    "scribedit_training_loss",
    "single_task_average",
    "stage_training_flags",
    "velocity_target",
    "whole_image_fm_loss",
]
