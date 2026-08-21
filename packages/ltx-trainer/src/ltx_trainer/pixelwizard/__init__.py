"""PixelWizard: efficient 2K/4K video generation (Li et al., arXiv:2605.25801).

Reference: anchor-guided injector, noise-span shortcut training, paper Tables 1–5.
Full Wan2.2 fine-tuning and UltraVideo training are external.
"""

from ltx_trainer.pixelwizard.anchor import degrade_anchor
from ltx_trainer.pixelwizard.config import PixelWizardConfig
from ltx_trainer.pixelwizard.injector import AnchorGuidedInjector
from ltx_trainer.pixelwizard.losses import flow_matching_loss, shortcut_training_loss
from ltx_trainer.pixelwizard.model import PixelWizardDiTBlock, PixelWizardVelocityHead
from ltx_trainer.pixelwizard.pipeline import (
    evaluation_demo,
    framework_card,
    table_ablation_aghs,
    table_ablation_inference_steps,
    table_ablation_shortcut,
    table_detail_quality,
    table_memory,
    table_vbench_hr,
    table_vbench_long,
    table_video_sr,
    training_step_demo,
)
from ltx_trainer.pixelwizard.shortcut import (
    calibration_weight,
    candidate_step_sizes,
    select_shortcut_step,
    shortcut_step,
)

__all__ = [
    "AnchorGuidedInjector",
    "PixelWizardConfig",
    "PixelWizardDiTBlock",
    "PixelWizardVelocityHead",
    "calibration_weight",
    "candidate_step_sizes",
    "degrade_anchor",
    "evaluation_demo",
    "flow_matching_loss",
    "framework_card",
    "select_shortcut_step",
    "shortcut_step",
    "shortcut_training_loss",
    "table_ablation_aghs",
    "table_ablation_inference_steps",
    "table_ablation_shortcut",
    "table_detail_quality",
    "table_memory",
    "table_vbench_hr",
    "table_vbench_long",
    "table_video_sr",
    "training_step_demo",
]
