"""Reliability-gated memory update (Sec. 2.3, Eq. 6–7)."""

from __future__ import annotations

import torch
from torch import Tensor


def masked_descriptor(features: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    """Eq. (6): target-specific descriptor via masked average pooling."""
    if features.dim() == 3:
        # (H, W, D)
        weights = mask.reshape(features.shape[0], features.shape[1], 1)
        num = (weights * features).sum(dim=(0, 1))
        den = weights.sum() + eps
        return num / den
    if features.dim() == 2 and mask.dim() == 2:
        weights = mask
        num = (weights.unsqueeze(-1) * features).sum(dim=(0, 1))
        den = weights.sum() + eps
        return num / den
    raise ValueError("features must be (H,W,D) and mask (H,W)")


def feature_consistency(descriptor: Tensor, anchor: Tensor) -> float:
    """Eq. (7): cosine consistency with first-frame anchor."""
    d = descriptor / descriptor.norm().clamp(min=1e-8)
    a = anchor / anchor.norm().clamp(min=1e-8)
    return float((d * a).sum())


def reliability_write_gate(consistency: float, tau: float) -> bool:
    """W_t = I[g_t > τ]."""
    return consistency > tau


def gate_memory_sequence(
    descriptors: list[Tensor],
    anchor: Tensor,
    tau: float,
) -> list[bool]:
    """Per-frame write decisions for a propagated sequence."""
    return [reliability_write_gate(feature_consistency(d, anchor), tau) for d in descriptors]
