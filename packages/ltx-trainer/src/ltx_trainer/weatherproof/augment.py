"""Weak / strong augmentations (Sec. 2.3–2.4)."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch import Tensor


def weak_augment(img: Tensor) -> Tensor:
    """Mild resize/crop/flip — preserves content for teacher pseudo labels."""
    x = img
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    _, _, h, w = x.shape
    scale = random.uniform(0.75, 1.0)
    nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
    x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    if random.random() < 0.5:
        x = torch.flip(x, dims=[-1])
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x.squeeze(0) if squeeze else x


def strong_augment(img: Tensor) -> Tensor:
    """Strong appearance + spatial perturbations for student branch."""
    x = weak_augment(img)
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    if random.random() < 0.5:
        x = x * random.uniform(0.6, 1.4)
    if random.random() < 0.3:
        x = x.mean(dim=1, keepdim=True).expand_as(x)
    if random.random() < 0.5:
        k = 5
        x = F.avg_pool2d(F.pad(x, (k // 2,) * 4, mode="reflect"), k, stride=1)
    x = x.clamp(0.0, 1.0)
    return x.squeeze(0) if squeeze else x
