"""GeM pooling and Hierarchical Aggregation Module (Sec. 4.3–4.5)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.hypevpr.config import HypeVPRConfig
from ltx_trainer.hypevpr.hierarchy import iter_level_groups, num_windows_at_level
from ltx_trainer.hypevpr.poincare import einstein_midpoint, expmap0


def gem_pool(feat: Tensor, p: float, eps: float = 1e-6) -> Tensor:
    """Generalized mean pooling over spatial dims (Eq. 8)."""
    if feat.dim() == 4:
        pooled = feat.clamp_min(eps).pow(p).mean(dim=(-2, -1)).pow(1.0 / p)
    else:
        pooled = feat
    return pooled


class LevelAggregator(nn.Module):
    """A_ℓ: GeM + linear projection (Eq. 8, 12)."""

    def __init__(self, in_dim: int, out_dim: int, gem_p: float) -> None:
        super().__init__()
        self.gem_p = gem_p
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, feat: Tensor) -> Tensor:
        return self.linear(gem_pool(feat, self.gem_p))


class HierarchicalAggregationModule(nn.Module):
    """HAM: level-wise Euclidean descriptors → hyperbolic tree (Eq. 13–15)."""

    def __init__(self, cfg: HypeVPRConfig, backbone_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.level_aggs = nn.ModuleList(
            [
                LevelAggregator(backbone_dim, cfg.descriptor_dim, cfg.gem_p)
                for _ in range(cfg.hierarchy_levels)
            ]
        )

    def forward(self, window_feats: list[Tensor]) -> dict[int, list[Tensor]]:
        """
        window_feats: finest-level features per window.
        Returns level → list of group hyperbolic descriptors h^(ℓ,k).
        """
        c = self.cfg.curvature
        l_total = self.cfg.hierarchy_levels
        n = len(window_feats)
        assert n == num_windows_at_level(l_total)

        leaf_euclid = [self.level_aggs[l_total - 1](f) for f in window_feats]
        leaf_hyp = [expmap0(d, c=c) for d in leaf_euclid]

        by_level: dict[int, list[Tensor]] = {l_total: leaf_hyp}

        for level in range(l_total - 1, 0, -1):
            groups: list[Tensor] = []
            for _, idxs in iter_level_groups(level, l_total):
                # Eq. 14–15: group finest-window descriptors h^(L),j at each level ℓ
                children = [leaf_hyp[j] for j in idxs]
                stacked = torch.stack(children, dim=0)
                groups.append(einstein_midpoint(stacked, c=c))
            by_level[level] = groups

        return by_level
