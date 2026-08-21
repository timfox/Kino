"""Synthetic FogAct-style paired clips."""

from __future__ import annotations

import random

import torch
from torch import Tensor

from ltx_trainer.fognet.classes import FOG_INTENSITIES, NUM_CLASSES
from ltx_trainer.fognet.fog import apply_fog


def _motion_frame(base: Tensor, t: int, total: int) -> Tensor:
    """Simple horizontal motion cue per action."""
    shift = int((t / max(total - 1, 1) - 0.5) * 8)
    return torch.roll(base, shifts=shift, dims=-1)


def _class_pattern(label: int, h: int, w: int) -> Tensor:
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    phase = (label % 7) / 7.0
    blob = ((xx - 0.5) ** 2 + (yy - 0.4) ** 2 < 0.08 + phase * 0.02).float()
    stripe = ((xx + yy + phase) % 0.3 < 0.12).float()
    rgb = torch.stack([blob * 0.9, stripe * 0.6 + phase * 0.2, (1 - blob) * 0.5], dim=0)
    return rgb.clamp(0, 1)


def synthesize_clip(
    *,
    label: int | None = None,
    frames: int = 8,
    size: int = 64,
    intensity: str | None = None,
) -> tuple[Tensor, Tensor, int]:
    """Return ``(clean_TCHW, foggy_TCHW, label)``."""
    label = random.randrange(NUM_CLASSES) if label is None else label
    intensity = random.choice(FOG_INTENSITIES) if intensity is None else intensity
    base = _class_pattern(label, size, size)
    clean = torch.stack([_motion_frame(base, t, frames) for t in range(frames)], dim=0)
    foggy = apply_fog(clean, intensity=intensity)
    return clean, foggy, label


def synthesize_batch(
    batch_size: int,
    *,
    frames: int = 8,
    size: int = 64,
) -> tuple[Tensor, Tensor, Tensor]:
    clips_c, clips_f, labels = [], [], []
    for _ in range(batch_size):
        c, f, y = synthesize_clip(frames=frames, size=size)
        clips_c.append(c)
        clips_f.append(f)
        labels.append(y)
    return (
        torch.stack(clips_c),
        torch.stack(clips_f),
        torch.tensor(labels, dtype=torch.long),
    )
