"""OIQA database cards (Sec. 4.1)."""

from __future__ import annotations

from typing import Any


def datasets_card() -> dict[str, Any]:
    return {
        "databases": [
            {"name": "CVIQ", "distorted": 528, "refs": 16, "distortion": "uniform (JPEG/H.264/H.265)"},
            {"name": "OIQA", "distorted": 320, "refs": 16, "distortion": "uniform (JPEG/J2K/blur/noise)"},
            {"name": "JUFE-10K", "images": 10320, "distortion": "non-uniform"},
            {"name": "OIQ-10K", "images": 10000, "distortion": "uniform + non-uniform"},
        ],
        "2d_adaptation": ["KADID-10K", "KonIQ-10K"],
        "split": "80% train / 20% test per database",
    }
