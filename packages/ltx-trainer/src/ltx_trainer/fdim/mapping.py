"""Component logistic mapping and fusion (Eq. 1–2)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class LogisticMapping:
    beta1: float = 1.0
    beta2: float = 0.1
    beta3: float = 50.0
    beta4: float = 0.0

    def __call__(self, q: Tensor) -> Tensor:
        b1, b2, b3, b4 = self.beta1, self.beta2, self.beta3, self.beta4
        return b1 * (0.5 - 1.0 / (1.0 + torch.exp(b2 * (q - b3)))) + b4


def fuse_component_scores(deep: Tensor, trad: Tensor) -> Tensor:
    return 0.5 * (deep + trad)
