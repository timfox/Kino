"""DL3DV-10K-Blur dataset card (Sec. III-B)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.deblur_nvs.config import (
    BLUR_WINDOW_SIZES,
    DL3DV_BLUR_PAIR_COUNT,
    DL3DV_IMAGES_PER_SCENE,
    DL3DV_SCENE_COUNT,
    INTERPOLATION_RATE,
)


def dataset_card() -> dict[str, Any]:
    return {
        "name": "DL3DV-10K-Blur",
        "source": "DL3DV-10K",
        "scenes": DL3DV_SCENE_COUNT,
        "images_per_scene": DL3DV_IMAGES_PER_SCENE,
        "sharp_blur_pairs": DL3DV_BLUR_PAIR_COUNT,
        "interpolation_rate": INTERPOLATION_RATE,
        "blur_window_sizes": list(BLUR_WINDOW_SIZES),
        "formation": "finite-exposure temporal average over interpolated sharp frames",
        "equation": "I_blur ≈ (1/N) Σ I_sharp_i, N ~ U{5,7,9,11}",
    }
