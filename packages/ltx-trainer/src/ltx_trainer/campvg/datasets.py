"""3D-FRONT panoramic trajectory dataset card (Sec. 4.1.1)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.campvg.benchmarks import DATASETS
from ltx_trainer.campvg.config import ERP_HEIGHT, ERP_WIDTH, NUM_FRAMES


def datasets_card() -> dict[str, Any]:
    return {
        "name": "3D-FRONT ERP trajectories",
        "scenes": DATASETS["scenes"],
        "frames_per_clip": NUM_FRAMES,
        "height": ERP_HEIGHT,
        "width": ERP_WIDTH,
        "augmentation": "random conditional frame (CamI2V-style)",
        "rendering": "cubemap → equirectangular",
    }
