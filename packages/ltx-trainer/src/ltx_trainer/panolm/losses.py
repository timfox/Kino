"""Training / judge losses for smoke (VQA uses GPT-score at eval)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def vqa_ce_loss(logits: Tensor, labels: Tensor) -> Tensor:
    return F.cross_entropy(logits, labels)


def contrastive_alignment_loss(vision: Tensor, text: Tensor, temperature: float = 0.07) -> Tensor:
    """Stub region-text alignment inspired by Sec. III-E RTC."""
    v = F.normalize(vision, dim=-1)
    t = F.normalize(text, dim=-1)
    logits = torch.matmul(v, t.transpose(-1, -2)) / temperature
    labels = torch.arange(v.shape[0], device=v.device)
    return F.cross_entropy(logits, labels)
