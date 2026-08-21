"""Channel-Aware Enhancement + quality regression (Sec. III-E)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CAEModule(nn.Module):
    """CAE on deepest stage feature F4 (Eq. 10)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.dropout = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.norm = nn.LayerNorm([channels])

    def forward(self, f4: Tensor) -> Tensor:
        mid = self.conv1(torch.relu(self.dw3(f4)))
        ref = self.dropout(self.conv2(self.dw5(mid)))
        out = f4.permute(0, 2, 3, 1)
        out = self.norm(out + ref.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out + f4


class QualityRegressor(nn.Module):
    """Concat Flatten(Fcae), Flatten(Faff) → FC (Eq. 11–12)."""

    def __init__(self, fcae_dim: int, faff_dim: int) -> None:
        super().__init__()
        in_dim = fcae_dim + faff_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, fcae: Tensor, faff: Tensor) -> Tensor:
        v = torch.cat([fcae.flatten(1), faff.flatten(1)], dim=1)
        return self.head(v).squeeze(-1)
