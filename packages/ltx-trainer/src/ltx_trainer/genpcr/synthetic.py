"""Synthetic point cloud pairs."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.genpcr.config import GenPcrConfig


def synthetic_point_cloud(cfg: GenPcrConfig, *, seed: int = 0) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    n = cfg.num_points
    return torch.randn(n, 3, generator=g) * 2.0 + torch.tensor([0.0, 0.0, 3.0])


def synthetic_pair(cfg: GenPcrConfig) -> tuple[Tensor, Tensor]:
    p = synthetic_point_cloud(cfg, seed=0)
    q = p + torch.randn_like(p) * 0.05
    q[:, 0] += 0.3
    return p, q
