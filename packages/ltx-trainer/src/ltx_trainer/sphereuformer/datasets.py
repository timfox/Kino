"""Evaluation datasets (Stanford2D3D, Structured3D)."""

from __future__ import annotations

from typing import Any


def datasets_card() -> dict[str, Any]:
    return {
        "train_eval": ["Stanford2D3D", "Structured3D"],
        "stanford2d3d": {
            "classes_seg": 13,
            "depth_cap_m": 10.0,
            "train_iters": 25000,
            "batch": 16,
        },
        "structured3d": {
            "classes_seg": 40,
            "depth_cap_m": 5.0,
            "train_iters": 160000,
            "batch": 16,
        },
        "optimizer": "Adam 1e-4",
        "protocol": "PanoFormer training config (supplement B)",
    }
