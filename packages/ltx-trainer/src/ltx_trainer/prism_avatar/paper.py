"""PrismAvatar framework card for agents and docs."""

from __future__ import annotations

from ltx_trainer.prism_avatar.benchmarks import benchmarks_bundle
from ltx_trainer.prism_avatar.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL


def framework_card() -> dict:
    return {
        "name": "PrismAvatar",
        "title": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "task": "Monocular Gaussian head avatar + glasses-free lenticular display",
        "contributions": [
            "PMV supervision from natural head turns at yaw bins {15°, 20°, 25°}",
            "Strict head-and-hair matte with alignment gates and contour losses",
            "32-view subpixel prism encoding to 4K autostereoscopic raster",
            "Subject-specific distilled RGB driver for 38+ FPS display path",
        ],
        "benchmarks": benchmarks_bundle(),
    }
