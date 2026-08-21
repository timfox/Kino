"""Segmentation losses (Sec. 4.2, Eq. 49)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def dice_loss(pred: Tensor, target: Tensor, *, eps: float = 1e-5) -> Tensor:
    pred = F.softmax(pred, dim=1)
    target_oh = F.one_hot(target.long(), num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
    inter = (pred * target_oh).sum(dim=(2, 3, 4))
    denom = pred.sum(dim=(2, 3, 4)) + target_oh.sum(dim=(2, 3, 4))
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def segmentation_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L_Seg = L_CE + L_Dice (Eq. 49)."""
    ce = F.cross_entropy(pred, target.long())
    return ce + dice_loss(pred, target)
