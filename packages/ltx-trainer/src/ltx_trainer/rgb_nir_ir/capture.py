"""RGB–NIR capture and flash isolation (Fig. 2)."""

from __future__ import annotations

import torch
from torch import Tensor


def isolate_nir_flash(nir_on: Tensor, nir_off: Tensor) -> Tensor:
    """I_NIR = I_NIR-on − I_NIR-off (ambient removal)."""
    if nir_on.shape != nir_off.shape:
        raise ValueError("nir_on and nir_off must have the same shape")
    return nir_on - nir_off


def hdr_bracket_mean(frames: list[Tensor]) -> Tensor:
    """Simple HDR merge: mean of bracketed exposures (stub)."""
    if not frames:
        raise ValueError("frames must be non-empty")
    return torch.stack(frames, dim=0).mean(dim=0)


def exposure_schedule_ms() -> list[int]:
    """Paper Fig. 2 timing sketch (RGB / NIR flash on/off)."""
    return [1, 8, 32, 100, 32, 1, 32, 1, 32, 1]
