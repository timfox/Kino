"""Label-free training data catalog (Sec. 4.1, supp. A.1)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mtpano.config import (
    BACKBONE,
    INPUT_SIZE,
    PATCHES_PER_PANO,
    PAPER_ARXIV,
    TRAIN_ITERS,
    TRAIN_PANORAMAS,
)

TRAINING_SOURCES: list[dict[str, Any]] = [
    {"name": "Structured3D", "count": 20_041},
    {"name": "SUN360", "count": 34_260},
    {"name": "Matterport3D", "count": 10_359},
    {"name": "DiT360 synthetic", "count": 76_224},
]

EVAL_BENCHMARKS: list[str] = [
    "Structured3D",
    "Stanford2D3D",
    "Matterport3D",
    "SynPASS",
    "Deep360",
    "PanoSUNCG",
]


def training_dataset_card() -> dict[str, Any]:
    return {
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "total_panoramas": TRAIN_PANORAMAS,
        "sources": TRAINING_SOURCES,
        "synthetic_pipeline": "DiT360 (FLUX.1-dev + panorama LoRA, 2048×1024)",
        "patches_per_panorama": PATCHES_PER_PANO,
        "patch_fov_deg": [80, 120],
        "patch_yaw_deg": [0, 360],
        "patch_pitch_deg": [-90, 90],
        "pseudo_label_models": {
            "semseg": "InternImage-H",
            "depth_normals": "MoGe-2",
        },
        "supervision": "patch-wise P2E/E2P (no full-stitch pseudo maps)",
        "input_resolution": list(INPUT_SIZE),
        "backbone": BACKBONE,
        "train_iters": TRAIN_ITERS,
        "eval_benchmarks": EVAL_BENCHMARKS,
    }
