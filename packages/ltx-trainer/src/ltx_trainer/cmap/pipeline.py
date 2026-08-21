"""Thin inference wiring: text routing + MPVTC + optional symmetric text gates (no CLIP / IAP train loop)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.cmap.config import CMAPConfig
from ltx_trainer.cmap.mpvtc import (
    calibrate_task_thresholds,
    joint_class_confidence,
    prompting_weight,
    task_confidence_from_joint,
)
from ltx_trainer.cmap.routing import route_task


def cmap_infer_batch_routed(
    v: Tensor,
    tau: Tensor,
    per_task_class_masks: list[Tensor],
    prototypes: Tensor,
    class_text: Tensor,
    cfg: CMAPConfig,
    theta_up_per_task: Tensor,
    theta_low_per_task: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Full routing + MPVTC for a batch (each sample picks its own task).

    ``v`` (B, d); ``tau`` (T, d); ``per_task_class_masks[t]`` (C,) bool over global class axis;
    ``prototypes`` / ``class_text`` (C, K, d) / (C, d); thresholds ``(T,)`` scalars per task.

    Returns ``(w, C, t_star)`` each (B,) except ``t_star`` long.
    """
    if theta_up_per_task.shape != theta_low_per_task.shape or theta_up_per_task.dim() != 1:
        raise ValueError("theta_up_per_task and theta_low_per_task must be 1D tensors of same length")
    t_star = route_task(v, tau)
    b = v.shape[0]
    device, dtype = v.device, v.dtype
    w_out = torch.empty(b, device=device, dtype=dtype)
    c_out = torch.empty(b, device=device, dtype=dtype)
    joint_all = joint_class_confidence(v, prototypes, class_text)
    for i in range(b):
        tid = int(t_star[i].item())
        m = per_task_class_masks[tid]
        c = task_confidence_from_joint(joint_all[i : i + 1], m, topk=cfg.top_class_scores)
        c_out[i] = c[0]
        w_out[i] = prompting_weight(c, theta_up_per_task[tid], theta_low_per_task[tid])[0]
    return w_out, c_out, t_star


def cmap_calibrate_thresholds_from_training(
    train_confidences: Tensor,
    cfg: CMAPConfig,
) -> tuple[Tensor, Tensor]:
    """Convenience: ``θ_up``, ``θ_low`` from 1D per-sample confidences for one task."""
    return calibrate_task_thresholds(train_confidences, cfg)


__all__ = [
    "cmap_calibrate_thresholds_from_training",
    "cmap_infer_batch_routed",
]
