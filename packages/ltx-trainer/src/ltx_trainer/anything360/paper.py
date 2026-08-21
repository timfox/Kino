"""Paper knowledge export."""

from __future__ import annotations

from typing import Any

from ltx_trainer.anything360.benchmarks import benchmarks_bundle
from ltx_trainer.anything360.config import (
    IMAGE_BACKBONE,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    PROJECT_URL,
    VIDEO_BACKBONE,
)
from ltx_trainer.anything360.datasets import datasets_card
from ltx_trainer.anything360.pipeline import evaluation_demo_run


def framework_card() -> dict[str, Any]:
    return {
        "title": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "project": PROJECT_URL,
        "authors": [
            "Ziyi Wu",
            "Daniel Watson",
            "Andrea Tagliasacchi",
            "David J. Fleet",
            "Marcus A. Brubaker",
            "Saurabh Saxena",
        ],
        "affiliation": "Google DeepMind / SFU / U of Toronto",
        "components": [
            "Sequence-concat DiT conditioning (geometry-free)",
            "Circular Latent Encoding (VAE seam fix)",
            "Canonical gravity-aligned ERP targets",
        ],
        "image_backbone": IMAGE_BACKBONE,
        "video_backbone": VIDEO_BACKBONE,
        "contributions": [
            "No camera metadata at inference — learns pers↔ERP from data",
            "CLE removes ERP boundary discontinuity in VAE latents",
            "SOTA image (Laval/SUN360) and video (Argus eval) vs projection baselines",
            "Zero-shot competitive FoV / pose from exhaustive pano2pers search",
        ],
        "datasets": datasets_card(),
        "benchmarks": benchmarks_bundle(),
    }


def evaluation_demo() -> dict[str, Any]:
    return {"package": "anything360", **evaluation_demo_run()}
