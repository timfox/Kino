"""360D / Matterport3D / Stanford2D3D card."""

from __future__ import annotations

from typing import Any


def datasets_card() -> dict[str, Any]:
    return {
        "360D": {"train": 35977, "resolution": "256×512 native, 512×1024 in paper"},
        "Matterport3D": {"train": 10800, "resolution": "512×1024"},
        "Stanford2D3D": {"panoramas": 1413, "resolution": "512×1024"},
        "depth_range_m": {"360D": "0.1–10", "indoor": "0.1–16"},
        "metrics": ["MAE", "MRE", "RMSE", "RMSE(log)", "δ1", "δ2", "δ3"],
    }
