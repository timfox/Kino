"""Evaluation metrics: δ_smooth, CLIP-Dir proxies (Sec. 4.3)."""

from __future__ import annotations

import torch
from torch import Tensor


def trajectory_lpips_distances(
    sequence: list[Tensor],
) -> list[float]:
    """Per-step feature distance along enhancement trajectory (δ_smooth building block)."""
    if len(sequence) < 2:
        return []
    dists: list[float] = []
    for i in range(len(sequence) - 1):
        a, b = sequence[i].float().flatten(), sequence[i + 1].float().flatten()
        dists.append(float((a - b).pow(2).mean().sqrt().item()))
    return dists


def delta_smooth(sequence: list[Tensor]) -> float:
    """δ_smooth: mean consecutive L2 feature distance (lower = smoother). KSlider-style proxy."""
    dists = trajectory_lpips_distances(sequence)
    if not dists:
        return 0.0
    return sum(dists) / len(dists)


def clip_direction_score(
    sequence: list[Tensor],
    *,
    dark_reference: Tensor | None = None,
) -> float:
    """CLIP-Dir proxy: correlation of brightness with strength index (higher = better).

    Real CLIP-Dir uses text embeddings; this stub uses mean luminance vs step index.
    """
    if len(sequence) < 2:
        return 0.0
    from ltx_trainer.controllight.retinex import luminance_y

    brightness = []
    for img in sequence:
        y = luminance_y(img.clamp(0, 1))
        brightness.append(float(y.mean().item()))
    n = len(brightness)
    idx = torch.arange(n, dtype=torch.float32)
    b = torch.tensor(brightness)
    idx_c = idx - idx.mean()
    b_c = b - b.mean()
    denom = (idx_c.pow(2).sum() * b_c.pow(2).sum()).sqrt().clamp(min=1e-8)
    return float((idx_c * b_c).sum() / denom)
