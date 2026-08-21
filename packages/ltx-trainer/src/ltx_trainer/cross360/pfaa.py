"""Progressive Feature Aggregation with Attention — Eq. (2), Fig. 3."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class AttentionCrossScale(nn.Module):
    """ACS block: channel attention on decoded features (Fig. 3)."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, feat: Tensor) -> Tensor:
        b, c, _, _ = feat.shape
        w = self.mlp(self.pool(feat).view(b, c))
        return w.view(b, c, 1, 1)


class ProgressiveFeatureAggregationAttention(nn.Module):
    """PFAA: progressive ACS from coarse to fine (Eq. 2)."""

    def __init__(self, channels_per_scale: list[int]) -> None:
        super().__init__()
        self.acs = nn.ModuleList([AttentionCrossScale(c) for c in channels_per_scale])
        self.fuse = nn.Conv2d(channels_per_scale[-1], channels_per_scale[-1], 3, padding=1)

    def forward(self, decoded_features: list[Tensor]) -> Tensor:
        if not decoded_features:
            raise ValueError("decoded_features must be non-empty")
        # Paper Eq. (2): aggregate from coarsest → finest; list is fine→coarse, so reverse.
        decoded_features = list(reversed(decoded_features))
        f_acs = self.acs[0](decoded_features[0]) * decoded_features[0]
        for s in range(1, len(decoded_features)):
            feat = decoded_features[s]
            f_acs_up = torch.nn.functional.interpolate(
                f_acs, size=feat.shape[-2:], mode="bilinear", align_corners=False
            )
            combined = feat + f_acs_up
            f_acs = self.acs[s](combined) * combined
        return self.fuse(f_acs)
