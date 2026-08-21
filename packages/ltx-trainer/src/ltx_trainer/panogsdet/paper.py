"""Paper integration card and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.panogsdet.benchmarks import PAPER_TITLE, benchmarks_bundle
from ltx_trainer.panogsdet.config import PAPER_ARXIV, PAPER_URL


def framework_card() -> dict[str, Any]:
    b = benchmarks_bundle()
    return {
        "name": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "problem": (
            "Single-view panoramic 3D detection suffers when depth→point-cloud pipelines "
            "use FPS/voxelization that breaks surface continuity."
        ),
        "method": {
            "depth_branch": "Frozen Panoformer-style ERP depth + semantic features",
            "lifting": "Per-pixel semantic 3D Gaussians from depth + features (Eq. 1–2)",
            "optimization": "Voxel semantic refine + center/covariance blocks (Eq. 3–5)",
            "supervision": "Cube-map semantic BCE (Eq. 6) + TR3D-style detection losses (Eq. 7)",
            "detection": "Gaussian-guided head on foreground Gaussians",
        },
        "results": {
            "mAP@25": b["table1_ap25"]["mAP@25"],
            "mAP@50": b["table2_ap50"]["mAP@50"],
            "fps": b["table3_resources"]["Ours_fps"],
        },
        "reference_metrics": b,
        "integration": (
            "GOPEX implements continuous semantic Gaussian geometry, compact trainable stubs, "
            "and Structured3D table anchors. Production training uses full Panoformer weights "
            "and sparse 3D conv as in the paper."
        ),
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.panogsdet.mock import evaluation_smoke

    return evaluation_smoke()
