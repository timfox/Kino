"""Synthetic compressed pairs for ranking training."""

from __future__ import annotations

import random

import torch
from torch import Tensor


def synthesize_distortion(ref: Tensor, severity: float) -> Tensor:
    """Apply blur + noise + blocking proxy."""
    import torch.nn.functional as F

    if ref.dim() == 3:
        ref = ref.unsqueeze(0)
    k = max(3, int(3 + severity * 8))
    if k % 2 == 0:
        k += 1
    blur = F.avg_pool2d(ref, k, stride=1, padding=k // 2)
    noise = torch.randn_like(ref) * (0.01 + severity * 0.08)
    block = ref
    if severity > 0.3:
        block = F.interpolate(
            F.interpolate(ref, scale_factor=0.5, mode="nearest"),
            size=ref.shape[-2:],
            mode="nearest",
        )
    out = (1.0 - severity) * ref + severity * (0.5 * blur + 0.3 * block + 0.2 * (ref + noise))
    return out.clamp(0.0, 1.0).squeeze(0)


def sample_training_pair(h: int = 128, w: int = 128) -> tuple[Tensor, Tensor, Tensor, float, float]:
    ref = torch.rand(3, h, w)
    sev = random.random()
    dist = synthesize_distortion(ref, sev)
    mos = 6.0 - 5.0 * sev + random.gauss(0, 0.2)
    sigma = 0.5 + 0.5 * random.random()
    return ref, dist, dist, mos, sigma
