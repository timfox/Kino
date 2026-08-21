"""FogAct triplet schema and manifest helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FogActTriplet:
    """``(Sf, Sc, L)`` as in Sec. 3.2."""

    foggy_path: Path
    clean_path: Path
    label: int
    scene: str = ""
    perspective: str = "front"
    fog_intensity: str = "light"


def fogact_stats() -> dict[str, int | float | str]:
    """Paper-reported dataset summary (Tab. 1, Fig. 4)."""
    return {
        "num_videos": 9724,
        "num_classes": 55,
        "num_scenes": 10,
        "resolution": "1920x1080",
        "fps": 25,
        "avg_duration_sec": 8.27,
        "train_split": 0.8,
        "perspectives": 4,
    }
