"""Paper integration card and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.panoworld.benchmarks import PAPER_TITLE, benchmarks_bundle
from ltx_trainer.panoworld.config import PAPER_ARXIV, PAPER_URL, PROJECT_URL


def framework_card() -> dict[str, Any]:
    b = benchmarks_bundle()
    t2 = b["table2_panospace"]
    return {
        "name": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "project_url": PROJECT_URL,
        "problem": (
            "MLLMs treat ERP panoramas as wide 2D images or fragmented perspective views, "
            "missing observer-centered spherical geometry and seam continuity."
        ),
        "method": {
            "taxonomy": (
                "Semantic anchoring, spherical grounding, reference-frame transformation, "
                "depth-aware 3D spatial reasoning"
            ),
            "data": "570K ERP corpus + verified metadata graph (detection + semantic re-ID + depth)",
            "model": "PanoWorld with Spherical Spatial Cross-Attention (SSCA) after patch embedding (Eq. 7–10)",
            "benchmark": "PanoSpace-Bench (2K questions, 8 categories)",
        },
        "results": {
            "PanoSpace_overall": t2["overall"],
            "PanoSpace_bfov_miou": t2["bfov_miou"],
            "Hstar_zero_shot": b["table3_hstar"]["zero_shot_overall"],
            "R2R_SR": b["table4_r2r_ce"]["SR"],
        },
        "reference_metrics": b,
        "integration": (
            "GOPEX implements ERP geometry, SSCA, metadata-graph stubs, PanoSpace metrics, "
            "and paper table anchors. Production training uses Qwen3.5-VL + full instruction corpus."
        ),
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.panoworld.mock import evaluation_smoke

    return evaluation_smoke()
