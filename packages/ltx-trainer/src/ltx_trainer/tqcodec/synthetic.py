"""Synthetic audio and dataset summary (Sec. 4.1)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.tqcodec.config import CLIP_SECONDS, SAMPLE_RATE
from ltx_trainer.tqcodec.metrics import table2_datasets


def synthetic_waveform(
    batch: int = 4,
    *,
    seconds: float = CLIP_SECONDS,
    sample_rate: int = SAMPLE_RATE,
    seed: int | None = None,
) -> Tensor:
    """Generate synthetic music-like waveform stub."""
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    t = torch.arange(0, int(seconds * sample_rate), dtype=torch.float32) / sample_rate
    # mix of harmonics
    waves = []
    for i in range(batch):
        f0 = 220.0 + i * 17.0
        w = sum(torch.sin(2 * 3.14159 * f0 * k * t) / k for k in range(1, 6))
        w = w + 0.05 * torch.randn(t.numel(), generator=gen)
        waves.append(w)
    return torch.stack(waves).unsqueeze(1)


def dataset_summary() -> dict:
    return {
        "sample_rate_hz": SAMPLE_RATE,
        "clip_seconds": CLIP_SECONDS,
        "datasets": table2_datasets(),
        "training": {"batch_size": 32, "iterations": 400_000, "hours_8xh20": 30},
    }
