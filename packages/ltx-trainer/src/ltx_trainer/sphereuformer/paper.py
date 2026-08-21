"""Framework summary."""

from __future__ import annotations

from typing import Any

from ltx_trainer.sphereuformer.config import (
    NUM_ENCODER_STAGES,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    TRAIN_ITERS_S2D3D,
    TRAIN_ITERS_S3D,
)
from ltx_trainer.sphereuformer.integration import proceduralsky_card


def framework_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "venue": "arXiv 2024 (Tel Aviv University)",
        "components": [
            "Icosphere / hexasphere graph (rank 7–8, ~164K–655K nodes)",
            "U-shaped Spherical Attention Modules (SAM / SLSA)",
            "Vertical global PE + 7×7 relative position bias",
            "Center pool down / nearest up on sphere ranks",
            f"{NUM_ENCODER_STAGES} encoder stages × 2 SAB, skip concat decoder",
        ],
        "tasks": ["monocular 360° depth", "semantic segmentation"],
        "training": {
            "stanford2d3d_iters": TRAIN_ITERS_S2D3D,
            "structured3d_iters": TRAIN_ITERS_S3D,
            "loss_depth": "Berhu",
            "loss_seg": "cross-entropy (ignore background)",
        },
        "integration": (
            "Dense sphere-native depth/semantics complement ERP HDR skies and "
            f"cubemap generators (CubeDiff); panorama lighting: {proceduralsky_card()['partner']}."
        ),
        "proceduralsky": proceduralsky_card(),
    }
