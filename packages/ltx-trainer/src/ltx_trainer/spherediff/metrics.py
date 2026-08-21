"""Panoramic metrics stubs (distortion, end continuity)."""

from __future__ import annotations

import torch
from torch import Tensor


def end_continuity_score(erp: Tensor) -> Tensor:
    """Left/right ERP seam L1 (lower is better)."""
    left = erp[..., :, :4]
    right = erp[..., :, -4:]
    return (left - right).abs().mean()


def seam_quality_score(erp: Tensor) -> Tensor:
    """Higher is better: inverse seam error."""
    return 1.0 / (1.0 + end_continuity_score(erp))
