"""Adaptive modality gating (Sec. 3.4, Eq. 5–7)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def adaptive_modality_gating(
    f_audio: Tensor,
    f_visual: Tensor,
    *,
    w_a: Tensor | None = None,
    w_v: Tensor | None = None,
) -> tuple[Tensor, float, float]:
    """
    Compute ``w_a, w_v`` via softmax on gate logits and fuse representations.

    Returns:
        f_fused, w_a, w_v
    """
    dim = f_audio.shape[-1]
    if w_a is None:
        w_a = torch.randn(dim, device=f_audio.device, dtype=f_audio.dtype) * 0.02
    if w_v is None:
        w_v = torch.randn(dim, device=f_visual.device, dtype=f_visual.dtype) * 0.02

    g_a = (f_audio * w_a).sum()
    g_v = (f_visual * w_v).sum()
    weights = F.softmax(torch.stack([g_a, g_v]), dim=0)
    w_a_val, w_v_val = float(weights[0]), float(weights[1])
    f_fused = w_a_val * f_audio + w_v_val * f_visual
    return f_fused, w_a_val, w_v_val
