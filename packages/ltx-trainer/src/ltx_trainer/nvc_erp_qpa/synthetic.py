"""Synthetic ERP frames for stub encode."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.nvc_erp_qpa.config import NvcErpQpaConfig


def synthetic_erp_frame(cfg: NvcErpQpaConfig, *, batch_size: int = 1) -> Tensor:
    h, w = cfg.erp_height, cfg.erp_width
    rows = torch.linspace(0, 1, h).view(1, 1, h, 1).expand(batch_size, 1, h, w)
    cols = torch.linspace(0, 1, w).view(1, 1, 1, w).expand(batch_size, 1, h, w)
    return (0.4 * rows + 0.3 * cols + 0.2 * torch.sin(cols * 6.28)).clamp(0, 1).expand(
        batch_size, 3, h, w
    )
