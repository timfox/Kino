"""Framework card."""

from __future__ import annotations

from typing import Any

from ltx_trainer.par.config import (
    CFG_SCALE,
    DEFAULT_PAD_RATIO,
    LAMBDA_CONSISTENCY,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    PROJECT_PAGE,
    TRAIN_ITERATIONS,
)


def framework_card() -> dict[str, Any]:
    return {
        "title": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "project": PROJECT_PAGE,
        "components": [
            "Masked autoregressive (MAR) on continuous VAE latents",
            "Unified text-to-panorama and panorama outpainting (Eq. 4–5)",
            "Cyclic translation consistency loss (Eq. 7)",
            "Dual-space circular padding pre/post VAE (Eq. 9)",
        ],
        "training": {
            "init": "NOVA",
            "iterations": TRAIN_ITERATIONS,
            "cfg": CFG_SCALE,
            "lambda_consistency": LAMBDA_CONSISTENCY,
            "pad_ratio": DEFAULT_PAD_RATIO,
        },
        "tasks": ["text-to-panorama", "panorama outpainting", "panoramic editing"],
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.par.pipeline import evaluation_demo_run

    return {"package": "par", **evaluation_demo_run()}
