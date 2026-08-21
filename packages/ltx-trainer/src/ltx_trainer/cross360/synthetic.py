"""Synthetic ERP batch for smoke training."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.cross360.config import Cross360Config


def synthetic_batch(cfg: Cross360Config | None = None, *, batch_size: int = 2) -> tuple[Tensor, Tensor]:
    cfg = cfg or Cross360Config(height=64, width=128)
    erp = torch.rand(batch_size, 3, cfg.height, cfg.width)
    depth = torch.rand(batch_size, 1, cfg.height, cfg.width) * cfg.max_depth_m
    return erp, depth
