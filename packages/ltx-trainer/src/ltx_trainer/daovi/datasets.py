"""ODV360 dataset card."""

from __future__ import annotations

from typing import Any

from ltx_trainer.daovi.config import (
    ODV360_FRAMES,
    ODV360_TEST,
    ODV360_TRAIN,
    ODV360_VAL,
    TRAIN_RESOLUTION,
)


def datasets_card() -> dict[str, Any]:
    h, w = TRAIN_RESOLUTION
    return {
        "name": "ODV360",
        "citation": "NTIRE 2023 360° challenge (Cao et al.)",
        "splits": {"train": ODV360_TRAIN, "val": ODV360_VAL, "test": ODV360_TEST},
        "frames_per_clip": ODV360_FRAMES,
        "native_resolution": "540x270",
        "paper_resolution": f"{w}x{h}",
        "mask_policy": "random moving multi-region masks (not static single blob)",
    }
