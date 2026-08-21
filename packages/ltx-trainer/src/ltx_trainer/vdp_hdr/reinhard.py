"""Reinhard tone map for 8-bit previews."""

from __future__ import annotations

import torch
from torch import Tensor


def reinhard_tone_map(linear_chw: Tensor, *, key: float = 0.18) -> Tensor:
    x = linear_chw.detach().float().clamp(min=0.0)
    lum = 0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2]
    scale = key / (lum.mean().clamp(min=1e-8))
    mapped = x * scale
    return (mapped / (1.0 + mapped)).clamp(0.0, 1.0)
