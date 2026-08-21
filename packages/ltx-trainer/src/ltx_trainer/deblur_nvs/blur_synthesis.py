"""Finite-exposure motion blur synthesis (Eq. 5–6, Sec. III-B)."""

from __future__ import annotations

import random

import torch
from torch import Tensor


def temporal_average_blur(sharp_frames: Tensor, window: int) -> Tensor:
    """Average ``window`` consecutive sharp frames → one blurred frame.

    Args:
        sharp_frames: ``(T, C, H, W)`` interpolated sharp sequence.
        window: exposure integration length N.
    """
    if window < 1 or sharp_frames.shape[0] < window:
        raise ValueError("invalid window for sharp frame sequence")
    acc = torch.zeros_like(sharp_frames[0])
    for i in range(window):
        acc = acc + sharp_frames[i]
    return acc / float(window)


def synthesize_blur_pair(
    sharp: Tensor,
    *,
    window_sizes: tuple[int, ...] = (5, 7, 9, 11),
    motion_amplitude: float = 0.08,
    seed: int | None = None,
) -> tuple[Tensor, Tensor, int]:
    """Build a sharp/blur pair from one sharp image via synthetic sub-frame motion.

    Interpolation is approximated by translating the sharp image along a random
    2D trajectory and averaging (CPU smoke; full DL3DV-10K-Blur uses 8× video interp).
    """
    rng = random.Random(seed)
    n = rng.choice(window_sizes)
    c, h, w = sharp.shape
    frames: list[Tensor] = []
    dx = motion_amplitude / max(n - 1, 1)
    dy = motion_amplitude / max(n - 1, 1)
    for i in range(n):
        shift_x = int(round((i - (n - 1) / 2) * dx * w))
        shift_y = int(round((i - (n - 1) / 2) * dy * h))
        frames.append(torch.roll(torch.roll(sharp, shifts=shift_x, dims=2), shifts=shift_y, dims=1))
    blur = temporal_average_blur(torch.stack(frames, dim=0), n)
    return sharp, blur, n
