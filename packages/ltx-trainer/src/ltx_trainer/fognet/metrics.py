"""Top-1 / Top-5 accuracy (Tab. 3, 5)."""

from __future__ import annotations

import torch
from torch import Tensor


def top1_accuracy(logits: Tensor, labels: Tensor) -> float:
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        labels = labels.unsqueeze(0) if labels.dim() == 0 else labels
    pred = logits.argmax(dim=-1)
    return float((pred == labels).float().mean())


def top5_accuracy(logits: Tensor, labels: Tensor) -> float:
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        labels = labels.unsqueeze(0) if labels.dim() == 0 else labels
    k = min(5, logits.shape[-1])
    topk = logits.topk(k, dim=-1).indices
    hits = (topk == labels.unsqueeze(-1)).any(dim=-1)
    return float(hits.float().mean())
