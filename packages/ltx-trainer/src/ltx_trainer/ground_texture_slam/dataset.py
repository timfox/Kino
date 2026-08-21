"""Dataset metadata (Sec. V-A)."""

from __future__ import annotations

DATASET_URL = "https://gitlab.com/riselab/multi-session-ground-texture-data"
CODE_URL = "https://gitlab.com/riselab/multi-session-ground-texture-slam"


def dataset_info() -> dict[str, int | float | str | tuple[int, int]]:
    return {
        "sessions": 5,
        "images_per_session_range": (225, 249),
        "image_size_hw": (711, 1266),
        "camera_height_m": 0.72,
        "camera": "Intel RealSense D435i",
        "ground_truth": "Qualisys motion capture (mm-level)",
        "wear_model": "tape overlay simulating surface wear",
        "dataset_url": DATASET_URL,
        "code_url": CODE_URL,
    }
