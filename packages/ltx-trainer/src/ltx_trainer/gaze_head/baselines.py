"""Deterministic baselines (Sec. 3.3)."""

from __future__ import annotations

import torch
from torch import Tensor


def constant_head_motion(gaze: Tensor, head0: Tensor | None = None) -> Tensor:
    """Fixed head pose for entire sequence; default uses first frame of gaze's paired head or zeros."""
    t = gaze.shape[0]
    if head0 is None:
        h = torch.zeros(2, dtype=gaze.dtype, device=gaze.device)
    else:
        h = head0[:2]
    return h.unsqueeze(0).expand(t, -1).clone()


def mirror_gaze_inputs(gaze: Tensor) -> Tensor:
    """Copy gaze pitch/yaw as head pose (Mirror Gaze Inputs baseline)."""
    return gaze.clone()
