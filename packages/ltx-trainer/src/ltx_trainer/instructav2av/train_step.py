"""Two-stage training strategy (Sec. 4.3, 5.1)."""

from __future__ import annotations

from enum import Enum
from typing import Any

import torch
from torch import Tensor

from ltx_trainer.instructav2av.backbone import DualStreamEditor
from ltx_trainer.instructav2av.config import InstructAV2AVConfig


class TrainingStage(str, Enum):
    VIDEO_ONLY = "video_only"
    AUDIO_ONLY = "audio_only"
    JOINT = "joint"


def training_schedule(cfg: InstructAV2AVConfig | None = None) -> dict[str, Any]:
    cfg = cfg or InstructAV2AVConfig()
    return {
        "stage1_video_steps": cfg.stage1_video_steps,
        "stage1_audio_steps": cfg.stage1_audio_steps,
        "stage2_joint_steps": cfg.stage2_joint_steps,
        "lr": cfg.lr,
        "optimizer": cfg.optimizer,
        "gpus": cfg.gpus,
        "note": "Stage 1 bypasses cross-modal attention; stage 2 enables full dual-stream",
    }


def instructav2av_training_step(
    z1_v: Tensor,
    z1_a: Tensor,
    zs_v: Tensor,
    zs_a: Tensor,
    inst_emb: Tensor,
    *,
    stage: TrainingStage = TrainingStage.JOINT,
    cfg: InstructAV2AVConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or InstructAV2AVConfig()
    model = DualStreamEditor(cfg=cfg, latent_channels=z1_v.shape[1])
    cross = stage == TrainingStage.JOINT
    out = model.training_step(z1_v, z1_a, zs_v, zs_a, inst_emb, cross_modal=cross)

    if stage == TrainingStage.VIDEO_ONLY:
        out["loss"] = cfg.lambda_video * (out["loss"] / (cfg.lambda_video + cfg.lambda_audio))
    elif stage == TrainingStage.AUDIO_ONLY:
        out["loss"] = cfg.lambda_audio * (out["loss"] / (cfg.lambda_video + cfg.lambda_audio))

    return {"loss": out["loss"], "stage": stage.value, "t": float(out["t"].item())}
