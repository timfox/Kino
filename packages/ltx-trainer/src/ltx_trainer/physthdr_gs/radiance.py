"""Physical radiance composition: g(La, Hr) and φ(La, l) (Sec. 4.1)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RadianceComposer(nn.Module):
    """MLP radiance composer g: (La, Hr) → per-Gaussian HDR color c (Eq. 8)."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
            nn.Softplus(),
        )

    def forward(self, la: Tensor, hr: Tensor) -> Tensor:
        """
        Args:
            la: ambient illumination [N, 3] or [B, N, 3]
            hr: hemispherical reflectance [N, 3] or [B, N, 3]
        Returns:
            HDR color c [..., 3]
        """
        x = torch.cat([la, hr], dim=-1)
        return self.net(x)


class IlluminationModulator(nn.Module):
    """Virtual illumination modulator φ: (La, l) → ˆLa (Eq. 10)."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
            nn.Softplus(),
        )

    def forward(self, la: Tensor, lighting_level: Tensor) -> Tensor:
        """
        Args:
            la: [N, 3] ambient illumination
            lighting_level: scalar or [N, 1] target lighting l
        """
        if lighting_level.dim() == 0:
            l = lighting_level.expand(la.shape[0], 1)
        elif lighting_level.dim() == 1:
            l = lighting_level.unsqueeze(-1)
        else:
            l = lighting_level
        if l.shape[0] != la.shape[0]:
            l = l.expand(la.shape[0], -1)
        x = torch.cat([la, l], dim=-1)
        return self.net(x)
