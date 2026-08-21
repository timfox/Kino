"""TADA loss smoke."""

from __future__ import annotations

from typing import Any

import numpy as np


def toy_target_pair(mode: str, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(64, 64), dtype=np.int16)
    den = raw.copy()
    if mode == "denoise":
        den = np.clip(raw.astype(np.int32) - 5, 0, 255).astype(np.int16)
    return raw, den


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    from ltx_trainer.tada.loss import tada_loss

    raw, den = toy_target_pair("denoise", seed=seed)
    losses = tada_loss(raw, den, emulated_source=den)
    return {"raw_mean": round(float(raw.mean()), 2), **{k: round(v, 4) for k, v in losses.items()}}
