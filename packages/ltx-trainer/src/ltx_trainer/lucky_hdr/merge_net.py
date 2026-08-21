"""Softmax-weighted convex merge (Eq. 6–9)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.lucky_hdr.features import merge_features


class MergeStage(nn.Module):
    """Predict per-pixel softmax weights over ``base`` and ``warped``."""

    def __init__(self, *, channels: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(8, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, 1),
        )

    def forward(
        self,
        base: Tensor,
        warped: Tensor,
        *,
        valid: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        psi_b = merge_features(base)
        psi_w = merge_features(warped)
        logits = self.net(torch.cat([psi_b, psi_w], dim=0).unsqueeze(0)).squeeze(0)
        if valid is not None:
            v = valid.unsqueeze(0) if valid.dim() == 2 else valid
            logits = logits.clone()
            logits[1] = logits[1] + torch.log(v.clamp(min=1e-3).squeeze(0))
        weights = F.softmax(logits, dim=0)
        merged = weights[0:1] * base + weights[1:2] * warped
        return merged.squeeze(0), weights[0:1], weights[1:2]
