"""Dense360-Bench metrics: mask IoU and caption phrase recall stubs."""

from __future__ import annotations

import torch
from torch import Tensor


def mask_iou(pred: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """Binary mask IoU for grounding evaluation."""
    if pred.shape != target.shape:
        raise ValueError("pred and target masks must match shape")
    pred_b = pred > 0.5
    tgt_b = target > 0.5
    inter = (pred_b & tgt_b).float().sum(dim=(-2, -1))
    union = (pred_b | tgt_b).float().sum(dim=(-2, -1))
    return inter / (union + eps)


def caption_phrase_recall(pred_text: str, key_phrases: list[str]) -> float:
    """Phrase-coverage recall (Fig. 4) — case-insensitive substring match."""
    if not key_phrases:
        return 1.0
    low = pred_text.lower()
    hits = sum(1 for p in key_phrases if p.lower() in low)
    return hits / len(key_phrases)
