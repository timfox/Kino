"""Location-dependent confidence MLP (Sec. 4.2, DUSt3R-style)."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class ConfidenceMLP(nn.Module):
    """g_φ : R³ → R for per-receiver confidence."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_rx: Tensor) -> Tensor:
        return self.net(x_rx)
