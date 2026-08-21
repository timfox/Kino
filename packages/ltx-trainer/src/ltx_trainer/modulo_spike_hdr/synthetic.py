"""Synthetic modulo bursts for training smoke (Sec. VI-A)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.modulo_spike_hdr.lar import modulo_image
from ltx_trainer.lucky_hdr.tonemap import tone_map_mu


def synthesize_modulo_pair(
    hdr: Tensor,
    *,
    period: float = 256.0,
    exposure_scale: float = 1.0,
    noise_std: float = 0.01,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (modulo_im normalized, hdr_linear, i_mu_gt)."""
    linear = (hdr * exposure_scale).clamp(min=0.0)
    if noise_std > 0:
        linear = linear + torch.randn_like(linear) * noise_std
    im = modulo_image(linear, period) / period
    i_mu_gt = tone_map_mu(hdr.clamp(min=0.0))
    return im.clamp(0.0, 1.0), linear, i_mu_gt
