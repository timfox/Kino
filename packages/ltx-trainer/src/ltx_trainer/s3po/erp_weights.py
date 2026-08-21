"""ERP latitude weights for WSS-L1 (Eq. 7, Fig. 12)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def wss_distortion_weights(height: int, width: int, device: torch.device | None = None) -> Tensor:
    """ψ_{i,j} = cos((i + 0.5 − height/2) π / height); constant along columns."""
    i = torch.arange(height, device=device, dtype=torch.float32)
    psi_row = torch.cos((i + 0.5 - height / 2.0) * math.pi / height).clamp_min(1e-6)
    return psi_row.view(1, 1, height, 1).expand(1, 1, height, width)
