"""Dataset cards (Sec. 5.1)."""

from __future__ import annotations

from typing import Any


def datasets_card() -> dict[str, Any]:
    return {
        "train": {
            "name": "Matterport3D",
            "erp_resolution": "640×320",
            "train_pairs": 44700,
            "test_pairs": 4575,
            "overlap_threshold": 0.30,
        },
        "test_only": {
            "name": "Stanford2D3D",
            "test_pairs": 3460,
            "overlap_threshold": 0.50,
        },
        "qualitative": ["EgoNeRF", "OmniPhotos"],
        "training": {"steps": 300000, "batch_size": 4, "optimizer": "AdamW"},
    }
