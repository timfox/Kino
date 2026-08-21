"""Non-speech noise bank augmentation stub (§3.2)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.seam.config import SeamConfig


def inject_noise_bank(
    speech: np.ndarray,
    noise: np.ndarray,
    *,
    overlap_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(speech)
    overlap = max(1, int(n * overlap_ratio))
    if len(noise) < overlap:
        noise = np.tile(noise, int(np.ceil(overlap / len(noise))))
    start = int(rng.integers(0, max(1, len(noise) - overlap)))
    out = speech.copy()
    clip = noise[start : start + overlap]
    out[:overlap] = 0.7 * out[:overlap] + 0.3 * clip[:overlap]
    return np.clip(out, -1.0, 1.0)


def augmentation_demo(*, seed: int = 0, cfg: SeamConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SeamConfig()
    rng = np.random.default_rng(seed)
    n = int(cfg.window_s * cfg.sample_rate_hz)
    speech = rng.normal(0, 0.1, size=n)
    noise = rng.normal(0, 0.02, size=n // 2)
    ratio = float(rng.uniform(cfg.noise_overlap_min, cfg.noise_overlap_max))
    mixed = inject_noise_bank(speech, noise, overlap_ratio=ratio, rng=rng)
    return {
        "noise_bank_hours": cfg.noise_bank_hours,
        "overlap_ratio": round(ratio, 3),
        "overlap_range": [cfg.noise_overlap_min, cfg.noise_overlap_max],
        "mixed_rms": float(np.sqrt(np.mean(mixed**2))),
        "breaks_clean_scripted_heuristic": True,
    }
