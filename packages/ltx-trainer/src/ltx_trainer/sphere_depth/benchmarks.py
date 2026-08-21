"""Reference metrics from Gazzeh et al. (arXiv:2604.23432)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.sphere_depth.config import (
    CODE_URL,
    HIGH_DEFORM_DEG,
    LARGE_POSE_GRID_DEG,
    NUM_IMAGES,
    NUM_LANDMARKS,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    SMALL_DEFORM_DEG,
    SMALL_POSE_GRID_DEG,
)

# Table 2 — learned λ and ε (meters) on gravity-aligned / small / high deformation
TABLE2_DEPTH_ERRORS: list[dict[str, Any]] = [
    {"model": "ACDNet", "lambda": 1.11, "gravity_aligned": 0.21, "small_def": 0.23, "high_def": 1.38},
    {"model": "DepthAnywhere", "lambda": 0.56, "gravity_aligned": 0.26, "small_def": 0.34, "high_def": 2.32},
    {"model": "BiFuse++", "lambda": 0.93, "gravity_aligned": 0.38, "small_def": 0.32, "high_def": 2.04},
    {"model": "SliceNet", "lambda": 0.92, "gravity_aligned": 0.41, "small_def": 0.65, "high_def": 2.09},
]

# Sec. 3.2 — Depth Anything on cubemap faces
DEPTH_ANYTHING_CUBEMAP = {
    "lambda": 1.38,
    "gravity_aligned": 1.72,
}

# Sec. 3.3 — ACDNet small-pose sensitivity (gravity-aligned baseline)
ACDNET_SMALL_POSE_BASELINE = 0.14
ACDNET_SMALL_POSE_INCREASE_PCT = (9, 19)

# Table 1 — model comparison axes
TABLE1_MODELS: list[dict[str, str]] = [
    {
        "model": "Depth Anything v2",
        "full_360": False,
        "architecture": "ViT + DPT",
        "spherical_geometry": "Cubemap / planar; no native ERP",
    },
    {
        "model": "DepthAnywhere",
        "full_360": True,
        "architecture": "Transformer teacher-student (DA v2 teacher)",
        "spherical_geometry": "Cube-face pseudo-labels + ERP rotation aug",
    },
    {
        "model": "BiFuse++",
        "full_360": True,
        "architecture": "Two-branch ResNet-34",
        "spherical_geometry": "ERP + cubemap feature alignment",
    },
    {
        "model": "ACDNet",
        "full_360": True,
        "architecture": "ResNet + ACDConv decoder",
        "spherical_geometry": "ACDConv + circular padding",
    },
    {
        "model": "SliceNet",
        "full_360": True,
        "architecture": "ResNet + vertical slices + ConvLSTM",
        "spherical_geometry": "Gravity-aligned vertical slices",
    },
]


def table2_by_model() -> dict[str, dict[str, float]]:
    return {r["model"]: r for r in TABLE2_DEPTH_ERRORS}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": {"arxiv": PAPER_ARXIV, "title": PAPER_TITLE, "url": PAPER_URL, "code": CODE_URL},
        "dataset": {
            "num_images": NUM_IMAGES,
            "num_landmarks": NUM_LANDMARKS,
            "small_deform_deg": SMALL_DEFORM_DEG,
            "high_deform_deg": HIGH_DEFORM_DEG,
        },
        "table1_models": TABLE1_MODELS,
        "table2_depth_errors": TABLE2_DEPTH_ERRORS,
        "depth_anything_cubemap": DEPTH_ANYTHING_CUBEMAP,
        "pose_grids": {
            "large_deg": list(LARGE_POSE_GRID_DEG),
            "small_deg": list(SMALL_POSE_GRID_DEG),
        },
        "acdnet_small_pose": {
            "baseline_epsilon": ACDNET_SMALL_POSE_BASELINE,
            "relative_increase_pct_range": ACDNET_SMALL_POSE_INCREASE_PCT,
        },
    }
