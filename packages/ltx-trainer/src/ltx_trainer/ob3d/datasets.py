"""Dataset specification card (Sec. 3.4, Table 2)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ob3d.config import (
    EVAL_COUNT,
    GITHUB_URL,
    IMAGES_PER_TRAJECTORY,
    KAGGLE_URL,
    OB3D_SCENES,
    SPARSE_POINTS,
    TRAIN_COUNT,
    eval_indices,
    train_indices,
)


def datasets_card() -> dict[str, Any]:
    indoor = [s for s, e in OB3D_SCENES.items() if e == "indoor"]
    outdoor = [s for s, e in OB3D_SCENES.items() if e == "outdoor"]
    return {
        "scenes": list(OB3D_SCENES.keys()),
        "num_scenes": len(OB3D_SCENES),
        "indoor_scenes": indoor,
        "outdoor_scenes": outdoor,
        "modalities": ["RGB PNG", "depth OpenEXR", "normal OpenEXR", "camera JSON", "sparse SfM points"],
        "image_size": "1600×800",
        "images_per_trajectory": IMAGES_PER_TRAJECTORY,
        "trajectories": ["egocentric", "non_egocentric"],
        "train_eval_split": {
            "train_indices": train_indices(),
            "eval_indices": eval_indices(),
            "train_count": TRAIN_COUNT,
            "eval_count": EVAL_COUNT,
        },
        "protocols": ["camera_pose_estimation", "novel_view_synthesis", "3d_reconstruction"],
        "sparse_points": SPARSE_POINTS,
        "download": {"kaggle": KAGGLE_URL, "github": GITHUB_URL},
        "license": "CC BY-NC-SA 4.0",
    }
