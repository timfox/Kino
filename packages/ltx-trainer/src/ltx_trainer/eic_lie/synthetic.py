"""Synthetic low/normal-light pairs + event voxels for training."""

from __future__ import annotations

import random

import torch
from torch import Tensor

from ltx_trainer.eic_lie.events import events_to_sbt_voxel, synthetic_events_from_image


def synthesize_low_light_pair(
    height: int = 128,
    width: int = 128,
    *,
    event_bins: int = 5,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Return ``(low_rgb, normal_rgb, event_voxel)`` each suitable for ``EicLie.forward``.

    - ``low_rgb``: ``[3,H,W]``
    - ``normal_rgb``: GT ``[3,H,W]``
    - ``event_voxel``: ``[Bins,H,W]``
    """
    device = device or torch.device("cpu")
    gt = torch.rand(3, height, width, device=device)
    # Retinex-style darkening
    illum = torch.rand(1, height, width, device=device) * 0.4 + 0.05
    low = (gt * illum).clamp(0, 1)
    noise = torch.randn_like(low) * 0.02
    low = (low + noise).clamp(0, 1)
    events = synthetic_events_from_image(gt, num_events=height * width // 4)
    voxel = events_to_sbt_voxel(events, height=height, width=width, num_bins=event_bins).to(device)
    # Normalize voxel for stability
    if voxel.abs().max() > 0:
        voxel = voxel / voxel.abs().max()
    return low, gt, voxel


def augment_pair(
    low: Tensor,
    gt: Tensor,
    voxel: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if random.random() < 0.5:
        low = torch.flip(low, [-1])
        gt = torch.flip(gt, [-1])
        voxel = torch.flip(voxel, [-1])
    gamma = random.uniform(0.85, 1.15)
    low = low.pow(gamma).clamp(0, 1)
    return low, gt, voxel
