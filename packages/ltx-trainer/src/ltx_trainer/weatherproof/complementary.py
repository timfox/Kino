"""Complementary channel dropout (Sec. 2.4)."""

from __future__ import annotations

import torch
from torch import Tensor


def complementary_dropout(feat1: Tensor, feat2: Tensor) -> tuple[Tensor, Tensor]:
    """Apply masks ``M`` and ``1-M`` with scaling factor 2."""
    c = feat1.shape[1]
    m = torch.rand(c, device=feat1.device) > 0.5
    m = m.view(1, c, 1, 1).float()
    return 2.0 * m * feat1, 2.0 * (1.0 - m) * feat2
