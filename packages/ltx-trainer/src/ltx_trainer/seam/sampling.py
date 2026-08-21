"""Seam-aware provenance sampling stub (§3.2)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.seam.config import SeamConfig


def sample_window_from_segment(
    *,
    segment_len: int,
    window_len: int,
    rng: np.random.Generator,
) -> int | None:
    if segment_len < window_len:
        return None
    return int(rng.integers(0, segment_len - window_len + 1))


def seam_aware_sample(
    *,
    provenance_id: str,
    segment_offset: int,
    segment_len: int,
    window_len: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Draw a window constrained to a single provenance segment."""
    rng = np.random.default_rng(seed)
    start = sample_window_from_segment(segment_len=segment_len, window_len=window_len, rng=rng)
    if start is None:
        return {
            "provenance_id": provenance_id,
            "valid": False,
            "pad_with_noise": True,
            "cross_recording_join": False,
        }
    return {
        "provenance_id": provenance_id,
        "valid": True,
        "offset": segment_offset + start,
        "length": window_len,
        "cross_recording_join": False,
        "pad_with_noise": False,
    }


def sampling_demo(*, seed: int = 0, cfg: SeamConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SeamConfig()
    window_len = int(cfg.window_s * cfg.sample_rate_hz)
    ok = seam_aware_sample(
        provenance_id="podcast_001",
        segment_offset=0,
        segment_len=window_len * 3,
        window_len=window_len,
        seed=seed,
    )
    short = seam_aware_sample(
        provenance_id="podcast_002",
        segment_offset=1000,
        segment_len=window_len // 2,
        window_len=window_len,
        seed=seed + 1,
    )
    return {
        "window_s": cfg.window_s,
        "single_provenance": ok["valid"] and not ok["cross_recording_join"],
        "short_segment_pads_noise": short["pad_with_noise"],
    }
