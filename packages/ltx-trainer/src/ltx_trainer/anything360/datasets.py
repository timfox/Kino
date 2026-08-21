"""Training / eval dataset cards (Appendix A.1)."""

from __future__ import annotations

from typing import Any

IMAGE_DATASETS = ("Polyhaven", "Humus", "Structured3D", "Pano360")
VIDEO_DATASET = "360-1M (Argus filter)"
EVAL_IMAGE = ("Laval Indoor", "SUN360")
EVAL_VIDEO_COUNT = 101

TRAINING_AUG = {
    "fov_deg_range": [30, 120],
    "pitch_deg_range": [-60, 60],
    "roll_deg_range": [-15, 15],
    "horizontal_roll_erp": True,
}


def datasets_card() -> dict[str, Any]:
    return {
        "image_train": list(IMAGE_DATASETS),
        "video_train": VIDEO_DATASET,
        "image_eval": list(EVAL_IMAGE),
        "video_eval_clips": EVAL_VIDEO_COUNT,
        "canonical_pipeline": ["COLMAP stabilize", "GeoCalib gravity align"],
        "training_augmentation": TRAINING_AUG,
    }
