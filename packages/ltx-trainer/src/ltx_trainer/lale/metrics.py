"""Segmentation metrics and efficiency proxies."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def dice_loss(logits: Tensor, targets: Tensor, num_classes: int) -> Tensor:
    """Multi-class Dice loss (paper training objective)."""
    probs = F.softmax(logits, dim=1)
    targets_oh = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * targets_oh).sum(dims)
    denom = probs.sum(dims) + targets_oh.sum(dims)
    dice = (2 * inter + 1) / (denom + 1)
    return 1 - dice.mean()


def f1_from_logits(logits: Tensor, targets: Tensor, num_classes: int) -> float:
    pred = logits.argmax(dim=1)
    f1s = []
    for c in range(num_classes):
        tp = ((pred == c) & (targets == c)).sum().item()
        fp = ((pred == c) & (targets != c)).sum().item()
        fn = ((pred != c) & (targets == c)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(sum(f1s) / max(1, len(f1s)) * 100)
