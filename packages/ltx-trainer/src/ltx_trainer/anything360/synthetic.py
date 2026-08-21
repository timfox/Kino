"""Synthetic perspective + ERP pairs."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.anything360.config import Anything360Config


def synthetic_pair(cfg: Anything360Config | None = None, *, batch_size: int = 2) -> tuple[Tensor, Tensor]:
    cfg = cfg or Anything360Config()
    pers = torch.rand(batch_size, 3, cfg.pers_height, cfg.pers_width)
    erp = torch.rand(batch_size, 3, cfg.erp_height, cfg.erp_width)
    return pers, erp
