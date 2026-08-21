"""PWC-Net inspired correlation layer (Sec. III-C.2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CorrelationLayer(nn.Module):
    """Channel-wise correlation between two feature maps."""

    def __init__(self, max_displacement: int = 4) -> None:
        super().__init__()
        self.max_displacement = max_displacement
        self.post = nn.Sequential(
            nn.Conv2d(81, 64, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((4, 8)),
            nn.Flatten(),
        )

    def forward(self, fa: Tensor, fb: Tensor) -> Tensor:
        b, c, h, w = fa.shape
        fa_n = F.normalize(fa, dim=1)
        fb_n = F.normalize(fb, dim=1)
        costs = []
        for dy in range(-self.max_displacement, self.max_displacement + 1):
            for dx in range(-self.max_displacement, self.max_displacement + 1):
                shifted = torch.roll(fb_n, shifts=(dy, dx), dims=(2, 3))
                costs.append((fa_n * shifted).sum(dim=1, keepdim=True))
        cost_vol = torch.cat(costs, dim=1)
        return self.post(cost_vol)
