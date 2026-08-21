"""Channel-wise multi-view self-distillation (Algorithm 1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def dino_cross_entropy(
    teacher_logits: Tensor,
    student_logits: Tensor,
    *,
    teacher_temp: float = 0.04,
    student_temp: float = 0.1,
) -> Tensor:
    """Symmetric distillation loss stub."""
    t = F.softmax(teacher_logits / teacher_temp, dim=-1)
    s = F.log_softmax(student_logits / student_temp, dim=-1)
    return -(t * s).sum(dim=-1).mean()


def multi_view_distill_loss(
    teacher_views: list[Tensor],
    student_views: list[Tensor],
) -> Tensor:
    """Match each teacher global view to all student views (Algorithm 1, line 7)."""
    losses = []
    for zt in teacher_views:
        for zs in student_views:
            if zs is zt:
                continue
            losses.append(dino_cross_entropy(zt.detach(), zs))
    return torch.stack(losses).mean() if losses else torch.zeros(())
