"""Paper integration card and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.h_omnistereo.benchmarks import PAPER_TITLE, benchmarks_bundle
from ltx_trainer.h_omnistereo.config import PAPER_ARXIV, PAPER_URL, PROJECT_URL


def framework_card() -> dict[str, Any]:
    b = benchmarks_bundle()
    t2 = b["table2_stereo"]["Ours"]
    return {
        "name": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "project_url": PROJECT_URL,
        "problem": (
            "Top-bottom equirectangular stereo has vertical epipolar lines but lacks large "
            "training data and distortion-robust monocular priors consistent across views."
        ),
        "method": {
            "dataset": "2.8M synthetic top-bottom ERP pairs (Isaac Sim, GRUtopia/HM3D + chaotic/realistic layouts)",
            "normal_prior": "Heading-aligned normals (longitude-varying frame, Eq. 2) + ray cross-attention",
            "stereo": "FoundationStereo-style side-tuning + attentive hybrid cost filter + ConvGRU disparity/uncertainty",
            "training": "3 stages: normal → frozen-prior disparity → uncertainty NLL",
        },
        "results_table2": {
            "3d60_mae": t2["3d60_mae"],
            "3d60_warp_mae": t2["3d60_warp_mae"],
            "mvs_gi_mae": t2["mvs_gi_mae"],
        },
        "reference_metrics": b,
        "integration": (
            "GOPEX implements heading-aligned geometry, compact trainable stubs, and paper table anchors. "
            "Full Isaac Sim dataset and released weights: upstream H-OmniStereo repo. "
            "Complements equirectangular HDR environment maps (e.g. proceduralsky.com) for lighting, not stereo geometry."
        ),
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.h_omnistereo.mock import evaluation_smoke

    return evaluation_smoke()
