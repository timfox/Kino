"""Dataset cards (PanoExplorer + baselines)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.panoworld_x.config import (
    FRAME_STRIDE_METERS,
    MIN_TRAJECTORY_METERS,
    PANOEXPLORER_SCENES,
    PANOEXPLORER_VIDEOS,
)


def datasets_card() -> dict[str, Any]:
    return {
        "primary": {
            "name": "PanoExplorer",
            "source": "Unreal Engine synthetic (504 scenes)",
            "videos": PANOEXPLORER_VIDEOS,
            "pairing": "panoramic video + 6-DoF exploration route",
            "min_trajectory_m": MIN_TRAJECTORY_METERS,
            "frame_stride_m": FRAME_STRIDE_METERS,
            "filtering": "Video-LLaMA3 auto + manual first-frame QA",
        },
        "baselines_compared": [
            "360DVD (WEB360)",
            "Imagine360",
            "GenEX",
            "CameraCtrl",
            "AC3D",
        ],
        "eval_split": "200 held-out clips from PanoExplorer",
    }
