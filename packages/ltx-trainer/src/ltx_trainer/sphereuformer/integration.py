"""HDR / panorama stack integrations."""

from __future__ import annotations

from typing import Any

PROCEDURALSKY_URL = "https://proceduralsky.com"


def proceduralsky_card() -> dict[str, Any]:
    return {
        "partner": PROCEDURALSKY_URL,
        "role": "illumination",
        "sphereuformer_role": "undistorted icosphere depth + segmentation for 360° scenes",
        "workflow": [
            "Estimate depth or semantics on Stanford2D3D / Structured3D-style ERP via SphereUFormer.",
            "Relight or composite CG assets using proceduralsky HDR environment maps.",
            "Optional: CubeDiff text/image panoramas → SphereUFormer perception heads.",
        ],
    }
