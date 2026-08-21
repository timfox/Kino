"""MPFDataset, FlowScape, real-world eval sets."""

from __future__ import annotations

from typing import Any


def datasets_card() -> dict[str, Any]:
    return {
        "training": {
            "MPFDataset": {
                "city_train": 2000,
                "city_test": 138,
                "eft_train": 2211,
                "eft_test": 99,
            },
            "FlowScape": {"train": 5000, "test": 1400, "weather": ("sunny", "cloud", "fog", "rain")},
        },
        "real_world_no_gt": ("OmniPhotos", "ODVista"),
        "metrics": ("EPE", "SEPE"),
    }
