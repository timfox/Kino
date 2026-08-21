"""Dataset cards (HM3D, 2D-3D-S)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fastvidar.benchmarks import DATASETS


def datasets_card() -> dict[str, Any]:
    return {
        "train_ablate": {
            "name": "HM3D",
            "scenes_train": 800,
            "scenes_test": 200,
            "groups_train": DATASETS["HM3D_train_groups"],
            "rig": "4-camera, FOV 160°–360°, randomized poses",
            "erp_resolution": "640×320",
        },
        "zero_shot_eval": {
            "name": "Stanford 2D-3D-S",
            "groups": DATASETS["2D3DS_zero_shot_groups"],
            "rig": "4 fisheye 220° FOV, 90° separation, baseline 20√2 mm",
            "fine_tune": False,
        },
        "preprocessing": "All fisheye → common ERP lattice (Sec. III-A)",
    }
