"""Electrovibration rendering (Eq. 4, Sec. 4.5)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def electrovibration_voltage(
    y: Tensor,
    *,
    gain: float = 1.0,
    carrier_hz: float = 7000.0,
    sample_rate: float = 20_000.0,
) -> Tensor:
    """V(t) = Vg * sqrt(y(t) + min|y|) * cos(2π f_c t) (Eq. 4)."""
    y = y - y.min()
    offset = y.abs().min()
    amp = torch.sqrt((y + offset).clamp(min=0.0))
    t = torch.arange(y.numel(), device=y.device, dtype=y.dtype) / sample_rate
    carrier = torch.cos(2.0 * math.pi * carrier_hz * t)
    return gain * amp * carrier
