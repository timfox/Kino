"""Synthetic WeatherProof-style paired samples."""

from __future__ import annotations

import random

import torch
from torch import Tensor

from ltx_trainer.weatherproof.classes import NUM_CLASSES


def _make_scene(h: int, w: int) -> tuple[Tensor, Tensor]:
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, h),
        torch.linspace(0, 1, w),
        indexing="ij",
    )
    sky = (yy < 0.35).long() * 9
    road = ((yy >= 0.65) & (xx > 0.2) & (xx < 0.8)).long() * 3
    building = ((yy >= 0.35) & (yy < 0.65) & (xx > 0.5)).long() * 8
    tree = ((yy >= 0.35) & (yy < 0.65) & (xx <= 0.5)).long() * 1
    mask = sky + road + building + tree
    mask = mask.clamp(0, NUM_CLASSES - 1)
    rgb = torch.zeros(3, h, w)
    for c in range(NUM_CLASSES):
        m = mask == c
        rgb[0] += m.float() * random.uniform(0.1, 0.9)
        rgb[1] += m.float() * random.uniform(0.1, 0.9)
        rgb[2] += m.float() * random.uniform(0.1, 0.9)
    return rgb.clamp(0, 1), mask


def degrade_weather(clean: Tensor) -> Tensor:
    """Fog/rain-like appearance change without changing labels."""
    fog = torch.rand(1, clean.shape[1], clean.shape[2]) * 0.4 + 0.3
    noisy = clean * 0.6 + fog
    if random.random() < 0.5:
        noisy = torch.roll(noisy, shifts=random.randint(-2, 2), dims=-1)
    return noisy.clamp(0, 1)


def synthesize_pair(*, size: int = 128) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(clean_rgb, degraded_rgb, mask)``."""
    clean, mask = _make_scene(size, size)
    degraded = degrade_weather(clean)
    return clean, degraded, mask
