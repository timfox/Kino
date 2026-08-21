"""mIoU and mDice (Table 1)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.weatherproof.classes import IGNORE_LABEL, NUM_CLASSES


def mean_iou(pred: Tensor, target: Tensor, *, num_classes: int = NUM_CLASSES) -> float:
    """Per-class IoU averaged over present classes."""
    if pred.dim() == 3:
        pred = pred.argmax(dim=0)
    valid = target != IGNORE_LABEL
    pred = pred[valid]
    tgt = target[valid]
    ious: list[float] = []
    for c in range(num_classes):
        p = pred == c
        t = tgt == c
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        if union > 0:
            ious.append(float((inter / union).item()))
    return sum(ious) / max(len(ious), 1)


def mean_dice(pred: Tensor, target: Tensor, *, num_classes: int = NUM_CLASSES) -> float:
    if pred.dim() == 3:
        pred = pred.argmax(dim=0)
    valid = target != IGNORE_LABEL
    pred = pred[valid]
    tgt = target[valid]
    dices: list[float] = []
    for c in range(num_classes):
        p = pred == c
        t = tgt == c
        inter = (p & t).sum().float()
        denom = p.sum().float() + t.sum().float()
        if denom > 0:
            dices.append(float((2.0 * inter / denom).item()))
    return sum(dices) / max(len(dices), 1)
