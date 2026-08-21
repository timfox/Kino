"""Multi-level Feature Fusion (Sec. 3.4, Fig. 5)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MultiLevelFeatureFusion(nn.Module):
    """Pixel-wise spatial attention over branch outputs."""

    def __init__(self, ch: int, num_branches: int = 3) -> None:
        super().__init__()
        self.num_branches = num_branches
        self.attn = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, num_branches, 1),
        )

    def forward(self, branch_feats: list[Tensor]) -> Tensor:
        if len(branch_feats) == 1:
            return branch_feats[0]
        stack = torch.stack(branch_feats, dim=1)  # B×K×C×H×W
        ref = branch_feats[0]
        w = torch.softmax(self.attn(ref), dim=1)  # B×K×H×W
        w = w.unsqueeze(2)
        return (w * stack).sum(dim=1)


def fuse_by_addition(branch_feats: list[Tensor]) -> Tensor:
    """Table 5 baseline: direct pixel-wise sum."""
    out = branch_feats[0]
    for f in branch_feats[1:]:
        out = out + f
    return out
