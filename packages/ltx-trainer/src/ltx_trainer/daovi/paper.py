"""Framework summary card."""

from __future__ import annotations

from typing import Any

from ltx_trainer.daovi.config import GEODESIC_THRESHOLD_DEG, PAPER_ARXIV, PAPER_TITLE


def framework_card() -> dict[str, Any]:
    return {
        "title": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "components": {
            "GFCIP": "geodesic bidirectional flow consistency in image space (ε={}°)".format(
                GEODESIC_THRESHOLD_DEG
            ),
            "ODAFP": "depth-assisted DCN feature propagation + ACDConv + DGG distortion weighting",
            "transformer": "mask-guided sparse video transformer (ProPainter-style stub)",
        },
        "baselines": ("FuseFormer", "STTN", "ProPainter"),
        "dataset": "ODV360",
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.daovi.pipeline import evaluation_demo_run

    return {"package": "daovi", **evaluation_demo_run()}
