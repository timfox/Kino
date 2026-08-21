"""Synthetic HST training batch."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.legs_over_arms.config import LegsOverArmsConfig
from ltx_trainer.legs_over_arms.skeleton import FeatureConfig, total_feature_dim


def synthetic_batch(
    cfg: LegsOverArmsConfig,
    feature_config: FeatureConfig = "K3D_L",
    *,
    batch_size: int = 2,
    device: torch.device | str | None = None,
) -> dict[str, Tensor]:
    dev = torch.device(device) if device is not None else torch.device("cpu")
    b, a = batch_size, cfg.num_agents
    t, f = cfg.history_steps, cfg.future_steps
    past = torch.randn(b, a, t, 2, device=dev)
    future = torch.randn(b, a, f, 2, device=dev)
    fdim = total_feature_dim(feature_config)
    pose = torch.randn(b, a, fdim, device=dev) if fdim else None
    return {"past_xy": past, "future_xy": future, "pose_feat": pose}
