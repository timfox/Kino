"""Multi-view ERP depth fusion (Sec. III-C, Eq. 14–15)."""

from __future__ import annotations

import torch
from torch import Tensor


def erp_mean_fusion(
    depths: Tensor,
    masks: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Eq. (14–15): masked mean over S views.

    depths: [S, H, W]; masks: [S, H, W] ∈ {0,1}.
  Returns fusion depth and coverage count.
    """
    weighted = depths * masks
    count = masks.sum(dim=0).clamp(min=1.0)
    fused = weighted.sum(dim=0) / count
    return fused, count


def erp_nearest_fusion(depths: Tensor, masks: Tensor) -> Tensor:
    """Pick first valid view per pixel (baseline)."""
    s, h, w = depths.shape
    out = torch.zeros(h, w, device=depths.device, dtype=depths.dtype)
    for i in range(s):
        m = masks[i] > 0
        out[m & (out == 0)] = depths[i][m & (out == 0)]
    return out


def erp_confidence_weighted_fusion(
    depths: Tensor,
    masks: Tensor,
    confidence: Tensor,
) -> Tensor:
    """Confidence-weighted variant (Table II)."""
    w = masks * confidence.clamp(min=1e-6)
    return (depths * w).sum(dim=0) / w.sum(dim=0).clamp(min=1e-6)
