"""Exposure-aware RoPE offset module (Tedla et al. — ready for LTX DiT wiring)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ExposureRoPEOffset(nn.Module):
    """Map per-frame EV + temporal index → rotary offset vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, ev: Tensor, frame_idx: Tensor, token_idx: Tensor) -> Tensor:
        feats = torch.stack([ev, frame_idx, token_idx], dim=-1).float()
        return self.net(feats)
