"""Torch-free evaluation_smoke for DIVA (import when torch unavailable)."""

from __future__ import annotations

import random


def evaluation_smoke(*, seed: int = 42) -> dict[str, object]:
    rng = random.Random(seed)
    ratio = 0.2 + 0.4 * rng.random()
    return {
        "batch_size": 4,
        "hidden_dim": 16,
        "mask_ratio": round(ratio, 4),
        "torch_available": False,
    }
