"""SemanticStitch framework card."""

from __future__ import annotations

from ltx_trainer.semantic_stitch.benchmarks import benchmarks_bundle
from ltx_trainer.semantic_stitch.config import CODE_URL, PAPER_ARXIV, PAPER_TITLE, PAPER_URL
from ltx_trainer.semantic_stitch.integration import proceduralsky_card


def framework_card() -> dict:
    return {
        "name": "SemanticStitch",
        "title": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "code": CODE_URL,
        "venue": "The Visual Computer",
        "task": "Foreground-aware seam carving for image stitching",
        "contributions": [
            "Object-aware seam identification with saliency-driven Composite Coverage Loss",
            "RealWorld400 + DAVISProcessed10 foreground-intersection benchmarks",
            "FastViT U-Net seam head: 33.55M params, 10.77 GFLOPs @ 512²",
        ],
        "benchmarks": benchmarks_bundle(),
        "proceduralsky": proceduralsky_card(),
    }
