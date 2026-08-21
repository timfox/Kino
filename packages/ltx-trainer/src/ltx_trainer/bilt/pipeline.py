"""BiLT training pipeline and paper reporting."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ltx_trainer.bilt.augment import apply_augmentation
from ltx_trainer.bilt.config import BiLTConfig, MAX_NOISE, MAX_SHIFT_BANDS, AUGMENT_RATIO
from ltx_trainer.bilt.losses import latent_loss, reconstruction_loss
from ltx_trainer.bilt.metrics import (
    mape,
    r2_score,
    table1_architecture,
    table2_training_phases,
    table3_performance,
    table_complexity,
)
from ltx_trainer.bilt.model import BiLTAutoencoder


def train_step(
    model: BiLTAutoencoder,
    x: Tensor,
    mu_a: Tensor,
    mu_s: Tensor,
    *,
    augment: bool = False,
) -> tuple[Tensor, dict[str, float]]:
    if augment:
        x = apply_augmentation(
            x, max_shift=MAX_SHIFT_BANDS, max_noise=MAX_NOISE, ratio=AUGMENT_RATIO
        )
    out = model(x)
    rec = reconstruction_loss(out["mu_a"], mu_a, out["mu_s"], mu_s)
    lat = latent_loss(out["decode_output"])
    loss = rec + lat
    with torch.no_grad():
        stats = {
            "loss": float(loss),
            "loss_rec": float(rec),
            "loss_latent": float(lat),
            "r2_mu_a": r2_score(mu_a, out["mu_a"]),
            "r2_mu_s": r2_score(mu_s, out["mu_s"]),
            "mape_mu_a": mape(mu_a, out["mu_a"]),
            "mape_mu_s": mape(mu_s, out["mu_s"]),
        }
    return loss, stats


def count_parameters(model: BiLTAutoencoder) -> int:
    return model.count_parameters()


def paper_report() -> dict[str, Any]:
    return {
        "architecture": table1_architecture(),
        "training_phases": table2_training_phases(),
        "performance": table3_performance(),
        "complexity": table_complexity(),
    }


def curriculum_phase(epoch: int) -> dict[str, float | int | str]:
    """Return phase metadata for epoch (1-indexed, total 8000)."""
    if epoch <= 1000:
        t = (epoch - 1) / max(999, 1)
        lr = 1e-3 * (1 - t) + 2e-5 * t
        return {"phase": 1, "lr": lr, "augment_strength": 0.0}
    if epoch <= 5000:
        t = (epoch - 1000) / 4000
        return {"phase": 2, "lr": 2e-5, "augment_strength": t}
    t = (epoch - 5000) / 3000
    lr = 2e-5 * (1 - t) + 2e-6 * t
    return {"phase": 3, "lr": lr, "augment_strength": 1.0}
