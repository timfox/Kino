"""Synthetic ERP / planar batches for smoke training."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.vuga.config import VUGAConfig


def synthetic_batch(
    cfg: VUGAConfig,
    batch_size: int = 4,
    *,
    device: torch.device | str | None = None,
) -> dict[str, Tensor]:
    dev = torch.device(device) if device is not None else None
    rgb = torch.rand(batch_size, cfg.in_channels, cfg.input_size, cfg.input_size, device=dev)
    mos = torch.rand(batch_size, device=dev) * 4.0 + 1.0
    return {"rgb": rgb, "mos": mos}
