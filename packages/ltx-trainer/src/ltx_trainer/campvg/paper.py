"""Framework card for agents."""

from __future__ import annotations

from typing import Any

from ltx_trainer.campvg.config import DOI, EPIPOLAR_SAMPLES_K, PAPER_ARXIV, PAPER_TITLE, PAPER_URL


def framework_card() -> dict[str, Any]:
    return {
        "title": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "doi": DOI,
        "venue": "SIGGRAPH Asia 2025",
        "components": [
            "Panoramic Plücker embedding (Eq. 3–5)",
            "Pose encoder → U-Net latent injection",
            "Spherical epipolar mask (Eq. 6–9, K samples)",
            "Spherical epipolar attention (Eq. 10)",
            "DynamiCrafter I2V base (frozen); train pose + epipolar only",
        ],
        "default_epipolar_k": EPIPOLAR_SAMPLES_K,
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.campvg.pipeline import evaluation_demo_run

    return {"package": "campvg", **evaluation_demo_run()}
