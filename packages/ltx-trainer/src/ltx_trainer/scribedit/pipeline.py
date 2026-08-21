"""Training-step glue for scribble-guided editing (Fig. 1–2)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.scribedit.config import ScribEditConfig
from ltx_trainer.scribedit.curriculum import TrainingStage, stage_training_flags
from ltx_trainer.scribedit.losses import scribedit_training_loss, velocity_target
from ltx_trainer.scribedit.mask import edit_region_mask, resize_mask_to_latent
from ltx_trainer.scribedit.mosaic import MosaicSample, format_multi_instruction, multi_task_mosaic


def scribedit_fm_step(
    v_pred: Tensor,
    z0: Tensor,
    z1: Tensor,
    source: Tensor,
    target: Tensor,
    *,
    stage: TrainingStage = TrainingStage.SYNTHETIC_COVERAGE,
    cfg: ScribEditConfig | None = None,
) -> Tensor:
    """One flow-matching step with stage-dependent edit-focused term."""
    cfg = cfg or ScribEditConfig()
    flags = stage_training_flags(stage, cfg=cfg)
    v_star = velocity_target(z1, z0)
    edit_mask = None
    if flags["use_edit_focused_loss"]:
        m_img = edit_region_mask(source, target, threshold=cfg.edit_mask_threshold)
        edit_mask = resize_mask_to_latent(m_img, v_pred.shape)
    return scribedit_training_loss(
        v_pred,
        v_star,
        edit_mask,
        edit_lambda=float(flags["edit_lambda"]),
        use_edit_focus=bool(flags["use_edit_focused_loss"]),
    )


def build_mosaic_training_sample(
    inputs: list[Tensor],
    targets: list[Tensor],
    instructions: list[str],
    *,
    layout: str = "1x2",
) -> tuple[MosaicSample, str]:
    """Construct mosaic composite and combined prompt."""
    sample = multi_task_mosaic(inputs, targets, instructions, layout=layout)
    prompt = format_multi_instruction(sample.instructions)
    return sample, prompt


def prepare_latent_masks(
    source: Tensor,
    target: Tensor,
    latent_shape: tuple[int, ...],
    *,
    cfg: ScribEditConfig | None = None,
) -> Tensor:
    """Image-space edit mask resized for latent FM (Stage 1 only)."""
    cfg = cfg or ScribEditConfig()
    m = edit_region_mask(source, target, threshold=cfg.edit_mask_threshold)
    return resize_mask_to_latent(m, latent_shape)
