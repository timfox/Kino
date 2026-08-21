"""CAST training pipeline and paper reporting."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ltx_trainer.cast.config import CASTConfig
from ltx_trainer.cast.losses import cast_loss, step_kl_loss
from ltx_trainer.cast.metrics import (
    table1_benchmarks,
    table2_average_ranks,
    table3_cast_wins,
    table4_ablation,
    table5_aliasing_experiment,
)
from ltx_trainer.cast.model import CAST
from ltx_trainer.cast.simplex import jensen_shannon, kl_divergence, normalize_simplex


def train_step(model: CAST, seq: Tensor) -> tuple[Tensor, dict[str, float]]:
    out = model(seq)
    target = out["target_seq"]
    pred = out["pred_seq"]
    # use last step operator stats for regularizer (representative)
    loss, stats = cast_loss(
        target.reshape(-1, target.shape[-1]),
        pred.reshape(-1, pred.shape[-1]),
        rho=out["rho_t"][:, -1],
        kernel_logits=out["kernel_logits"][:, -1],
        anchor=out["anchor"][:, -1],
        transported=out["transported"][:, -1],
        cfg=model.cfg,
    )
    with torch.no_grad():
        stats["kl_mean"] = float(kl_divergence(target, pred).mean())
        stats["jsd_mean"] = float(jensen_shannon(target, pred).mean())
    return loss, stats


def persistence_baseline(seq: Tensor) -> Tensor:
    return seq[:, :-1]


def count_parameters(model: CAST) -> int:
    return model.count_parameters()


def paper_report() -> dict[str, Any]:
    return {
        "benchmarks": table1_benchmarks(),
        "average_ranks": table2_average_ranks(),
        "cast_wins": table3_cast_wins(),
        "ablation": table4_ablation(),
        "aliasing_experiment": table5_aliasing_experiment(),
    }


def aliasing_js_lower_bound(u_up: Tensor, u_down: Tensor) -> float:
    """Theorem 1: JS_{1/2}(u_up, u_down) for equal-probability regimes."""
    return float(jensen_shannon(u_up.unsqueeze(0), u_down.unsqueeze(0)).item())
