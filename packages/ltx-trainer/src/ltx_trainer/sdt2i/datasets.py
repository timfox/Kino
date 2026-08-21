"""DSynView dataset card (Sec. 4.1)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.sdt2i.benchmarks import DSYNVIEW_PANORAMAS, DSYNVIEW_SCENES, DSYNVIEW_SEEDS


def datasets_card() -> dict[str, Any]:
    return {
        "name": "Dense-Synthetic-View (DSynView)",
        "scenes": DSYNVIEW_SCENES,
        "seeds_per_scene": DSYNVIEW_SEEDS,
        "test_panoramas": DSYNVIEW_PANORAMAS,
        "background_prompts": 3,
        "foreground_per_scene": "up to 3 masks",
        "mask_sizes": ["small", "medium", "large"],
        "mask_types": ["regular", "ERP-reprojected"],
        "metrics": ["IoU", "ImageReward", "FID", "CLIP-score", "CMMD"],
        "reference_images": 18144,
    }
