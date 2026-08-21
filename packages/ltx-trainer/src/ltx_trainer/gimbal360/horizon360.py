"""Horizon360 dataset card (Sec. 4, Appendix B)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.gimbal360.config import (
    ERP_TRAIN_SIZE,
    HORIZON360_SIZE,
    PERSPECTIVE_PER_PANO,
    PAPER_ARXIV,
)


def dataset_card() -> dict[str, Any]:
    """Metadata for gravity-aligned Horizon360 curation (no Hub download)."""
    return {
        "name": "Horizon360",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "size": HORIZON360_SIZE,
        "erp_resolution": list(ERP_TRAIN_SIZE),
        "perspective_variations_per_pano": PERSPECTIVE_PER_PANO,
        "canonicalization": (
            "Vanishing-point / zenith alignment: pitch and roll zeroed so the "
            "environment horizon lies on the ERP equator and verticals are plumb."
        ),
        "sources": [
            "Matterport3D",
            "Structured3D",
            "Poly Haven HDRIs",
            "SPEC / LayerPano3D-style web collections",
            "Unreal Engine renders",
        ],
        "pose_sampling": {
            "mixture_lambda": 0.7,
            "pitch": "70% N(0, 15°) + 30% U(-45°, 45°)",
            "roll": "80% N(0, 5°) + 20% U(-45°, 45°)",
            "fov": "50% N(60°, 10°) + 20% U(45°, 100°) (vertical FOV + aspect)",
            "yaw": "U(-π, π); ERP roll-centered so NFoV sits at ψ=0",
        },
        "eval_holdout": {
            "Structured3D": 500,
            "CVRG-Pano": 500,
            "total": 1000,
        },
        "gopex_note": "Stub only — use project page weights + Flux-fill for production training.",
    }
