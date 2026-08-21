"""PU21 preprocessing for HDR VQA (Sec. III-D)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.hdr_ingest import pu21_forward_linear_abs


def hdr_rgb_to_pu21_input(rgb: Tensor, *, l_peak: float = 1000.0) -> Tensor:
    """Display RGB ``[B,3,H,W]`` or ``[3,H,W]`` in ``[0,1]`` → PU21-normalized input."""
    squeeze = False
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)
        squeeze = True
    lin = rgb.clamp(0.0, 1.0) * l_peak
    out = pu21_forward_linear_abs(lin)
    return out.squeeze(0) if squeeze else out
