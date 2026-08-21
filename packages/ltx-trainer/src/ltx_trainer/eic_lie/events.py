"""Event representation: SBT voxel stacking (Sec. 3.1)."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


EventTuple = tuple[int, int, float, int]  # x, y, t, polarity ±1


def synthetic_events_from_image(
    image: Tensor,
    *,
    num_events: int = 500,
    threshold: float = 0.05,
) -> list[EventTuple]:
    """
    Generate pseudo-events from RGB frame differences vs blurred reference.

    Used for training/synthetic benchmarks when no DVS stream is available.
    """
    if image.dim() == 3:
        gray = image.mean(dim=0)
    else:
        gray = image.mean(dim=1).squeeze(0)
    h, w = gray.shape
    device = gray.device
    ref = torch.nn.functional.avg_pool2d(gray.unsqueeze(0).unsqueeze(0), 5, stride=1, padding=2).squeeze()
    diff = (gray - ref).abs()
    flat = diff.flatten()
    probs = flat / (flat.sum() + 1e-8)
    idx = torch.multinomial(probs, min(num_events, flat.numel()), replacement=True)
    events: list[EventTuple] = []
    for k, flat_i in enumerate(idx.tolist()):
        y, x = divmod(flat_i, w)
        pol = 1 if gray[y, x] > ref[y, x] else -1
        if float(diff[y, x]) < threshold:
            continue
        t = float(k) / max(num_events, 1)
        events.append((int(x), int(y), t, pol))
    if not events:
        events.append((w // 2, h // 2, 0.0, 1))
    return events


def events_to_sbt_voxel(
    events: Sequence[EventTuple],
    *,
    height: int,
    width: int,
    num_bins: int = 5,
    t0: float | None = None,
    t1: float | None = None,
) -> Tensor:
    """
    Stacking-Based on Time (SBT) polarity accumulation (Eq. 4).

    Returns voxel ``[num_bins, H, W]``.
    """
    voxel = torch.zeros(num_bins, height, width)
    if not events:
        return voxel
    times = [e[2] for e in events]
    t0 = float(min(times)) if t0 is None else t0
    t1 = float(max(times)) if t1 is None else t1
    span = max(t1 - t0, 1e-6)
    for x, y, t, p in events:
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        bin_idx = min(num_bins - 1, int((t - t0) / span * num_bins))
        voxel[bin_idx, y, x] += float(p)
    return voxel
