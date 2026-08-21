"""ODISR datasets (Sec. 4.1.1)."""

from __future__ import annotations

from typing import Any

TRAIN_DATASETS = ("ODI-SR", "Flickr360")
TEST_DATASETS = ("ODI-SR-test", "SUN360-Panorama", "Flickr360-val")
HR_SIZE = (1024, 2048)


def datasets_card() -> dict[str, Any]:
    return {
        "train": {
            "ODI-SR": {"images": 1200, "hr_size": HR_SIZE},
            "Flickr360": {"images": 3000, "hr_size": HR_SIZE},
        },
        "test": {
            "ODI-SR": 100,
            "SUN360-Panorama": 100,
            "Flickr360-val": 50,
        },
        "lr_generation": "fisheye downsampling (OSRT protocol)",
        "scales": [4, 8, 16],
        "patch_train": 256,
    }
