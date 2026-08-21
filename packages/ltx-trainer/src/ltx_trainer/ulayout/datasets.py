"""Dataset cards (Sec. 4.2)."""

from __future__ import annotations

from typing import Any


def datasets_card() -> dict[str, Any]:
    return {
        "panorama": [
            {
                "name": "PanoContext",
                "hub": None,
                "splits": {"train": 413, "val": 46, "test": 53},
            },
            {
                "name": "Stanford2D3D",
                "hub": None,
                "splits": {"train": 404, "val": 33, "test": 113},
            },
            {
                "name": "MatterportLayout",
                "hub": None,
                "splits": {"train": 1647, "val": 190, "test": 458},
                "layouts": 2295,
            },
        ],
        "perspective": [
            {
                "name": "LSUN",
                "train": 3880,
                "val": 389,
                "note": "ceiling+floor boundaries; pitch from horizon or Perspective Fields",
            },
        ],
    }
