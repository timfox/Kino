"""Paper stub smoke for EIC-LIE."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.eic_lie.losses import EicLieLoss
from ltx_trainer.eic_lie.metrics import psnr, ssim
from ltx_trainer.eic_lie.model import EicLie, EicLieConfig
from ltx_trainer.eic_lie.pipeline import enhance_low_light
from ltx_trainer.eic_lie.synthetic import synthesize_low_light_pair


def evaluation_smoke() -> dict[str, Any]:
    model = EicLie(EicLieConfig(base_channels=16, eici_stages=(1, 1), iaef_lite=True))
    loss_fn = EicLieLoss()
    low, gt, voxel = synthesize_low_light_pair(64, 64)
    pred = enhance_low_light(model, low, voxel)
    loss, stats = loss_fn(pred, gt)
    return {
        "package": "eic_lie",
        "paper": "arXiv:2605.22186",
        "enhanced_shape": list(pred.shape),
        "loss_total": round(float(stats["loss_total"]), 4),
        "psnr": round(psnr(pred.unsqueeze(0), gt.unsqueeze(0)), 2),
        "ssim": round(ssim(pred.unsqueeze(0), gt.unsqueeze(0)), 4),
        "loss_scalar": round(float(loss), 4),
    }
