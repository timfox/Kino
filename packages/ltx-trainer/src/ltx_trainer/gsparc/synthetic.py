"""Synthetic RF spectrum / CSI samples."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.gsparc.config import GSpaRCConfig


def synthetic_spectrum(cfg: GSpaRCConfig) -> Tensor:
    h, w = cfg.spectrum_height, cfg.spectrum_width
    r = torch.randn(h, w) * 0.1
    i = torch.randn(h, w) * 0.1
    return torch.stack([r, i], dim=0)


def synthetic_channel() -> Tensor:
    return torch.tensor([0.5, -0.2])


def synthetic_rx_position() -> Tensor:
    return torch.tensor([1.0, 2.0, 0.5])
