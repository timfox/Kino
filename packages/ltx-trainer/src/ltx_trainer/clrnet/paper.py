"""Paper integration card and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.clrnet.benchmarks import benchmarks_bundle, table2_clrnet_full, table3_clrnet_median
from ltx_trainer.clrnet.config import CODE_URL, PAPER_ARXIV, PAPER_TITLE, PAPER_URL


def framework_card() -> dict[str, Any]:
    t2 = table2_clrnet_full()
    t3 = table3_clrnet_median()
    return {
        "name": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "code_url": CODE_URL,
        "problem": (
            "Targetless extrinsic calibration for camera, lidar, and 4D radar; "
            "radar sparsity and lack of shared visual features make pairwise CR hard."
        ),
        "method": {
            "projection": "Equirectangular depth images (360° FoV) for lidar/radar",
            "architecture": "ResNet-18 encoders + PWC-style correlation + shared-feature regression",
            "loss": "Pairwise param/point losses + joint loop closure (λ=0.25)",
            "variants": "CRNet (camera-radar), CLRNet+4 (4-frame rigid)",
        },
        "results": {
            "VoD_ablation_MAE_cm": t2["transl_cm"],
            "VoD_CLRNet_median_cr_cm": t3["cr_transl_median"],
            "VoD_CLRNet_median_cl_cm": t3["cl_transl_median"],
            "CLRNet_plus4_rigid_cr_cm": 0.9,
        },
        "reference_metrics": benchmarks_bundle(),
        "integration": (
            "GOPEX implements ERP projection, CLRNet/CRNet stubs, and Tables II–IV anchors. "
            "Production weights: github.com/tudelft-iv upon release."
        ),
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.clrnet.mock import evaluation_smoke

    return evaluation_smoke()
