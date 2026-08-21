"""Datasets for PAR experiments (Sec. 4.1)."""

from __future__ import annotations

from typing import Any


def datasets_card() -> dict[str, Any]:
    return {
        "Matterport3D": {
            "split": "PanFusion train/val split",
            "resolution": "512×1024",
            "captions": "Janus-Pro-7B",
        },
        "SUN360": {"use": "OOD outpainting eval"},
        "Structured3D": {"use": "additional T2P ablation (9k/1k split)"},
    }
