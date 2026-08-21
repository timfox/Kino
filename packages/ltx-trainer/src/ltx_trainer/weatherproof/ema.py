"""EMA teacher update (Sec. 2.6)."""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def update_ema_teacher(teacher: nn.Module, student: nn.Module, *, decay: float = 0.999) -> None:
    """EMA on decoder weights only (DINOv2 backbone stays frozen)."""
    if hasattr(teacher, "decoder") and hasattr(student, "decoder"):
        t_params = list(teacher.decoder.parameters())
        s_params = list(student.decoder.parameters())
    else:
        t_params = list(teacher.parameters())
        s_params = list(student.parameters())
    for pt, ps in zip(t_params, s_params, strict=True):
        pt.data.mul_(decay).add_(ps.data, alpha=1.0 - decay)
