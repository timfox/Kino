"""Reference facts — Sphere360 (omniaudio/Sphere360)."""

from __future__ import annotations

from typing import Any

PAPER_HUB = "omniaudio/Sphere360"
SPHERE360_CLIP_DURATION_S = 10.0
SPHERE360_NUM_CLIPS = 103_000


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "hub_dataset": PAPER_HUB,
        "clip_duration_s": SPHERE360_CLIP_DURATION_S,
        "num_clips_approx": SPHERE360_NUM_CLIPS,
        "modalities": ["equirect_video", "foa_audio", "text"],
    }
