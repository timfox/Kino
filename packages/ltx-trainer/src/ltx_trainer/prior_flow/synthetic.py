"""Synthetic ERP frame pairs for CPU smoke."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.prior_flow.config import PriorFlowConfig


def synthetic_erp_pair(cfg: PriorFlowConfig) -> tuple[Tensor, Tensor]:
    """[1,3,H,W] consecutive frames with horizontal shift."""
    h, w = cfg.height, cfg.width
    yy = torch.linspace(0, 1, h).view(1, 1, h, 1).expand(1, 1, h, w)
    xx = torch.linspace(0, 1, w).view(1, 1, 1, w).expand(1, 1, h, w)
    f1 = torch.cat([xx, yy, torch.full_like(xx, 0.5)], dim=1)
    f2 = torch.roll(f1, shifts=3, dims=3)
    return f1, f2


def synthetic_flow_gt(cfg: PriorFlowConfig) -> Tensor:
    """Ground-truth primitive flow stub [1,2,H,W]."""
    flow = torch.zeros(1, 2, cfg.height, cfg.width)
    flow[:, 0] = 3.0
    return flow
