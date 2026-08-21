"""BerHu loss (Eq. 4–5)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.spherefusion.config import BERHU_T


def berhu_loss(pred: Tensor, target: Tensor, *, t: float = BERHU_T) -> Tensor:
    diff = (pred - target).abs()
    mask = diff < t
    l1 = diff[mask].sum()
    l2 = ((diff[~mask].pow(2) + t * t) / (2 * t)).sum()
    n = pred.numel()
    return (l1 + l2) / max(n, 1)
