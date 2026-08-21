"""Framework card."""

from __future__ import annotations

from typing import Any

from ltx_trainer.spherediff.config import (
    FOV_DEGREES,
    NUM_SPHERICAL_LATENTS,
    NUM_VIEW_DIRECTIONS,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    VIEW_OVERLAP,
)


def framework_card() -> dict[str, Any]:
    return {
        "title": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "components": [
            "Fibonacci spherical latent S on S² (Eq. 1–2)",
            "Spherical MultiDiffusion Ψ_S (Eq. 4–7)",
            "Dynamic latent sampling (Algorithm 1)",
            "Distortion-aware weighted averaging (Eq. 9–10)",
            "Multi-prompt elevation conditioning (Sec. 3.5)",
        ],
        "hyperparameters": {
            "num_latents": NUM_SPHERICAL_LATENTS,
            "num_views": NUM_VIEW_DIRECTIONS,
            "fov_deg": FOV_DEGREES,
            "overlap": VIEW_OVERLAP,
        },
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.spherediff.pipeline import evaluation_demo_run

    return {"package": "spherediff", **evaluation_demo_run()}
