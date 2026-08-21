"""Paper integration card and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.vuga.benchmarks import PAPER_TITLE, TABLE2_OIQA_VUGA, benchmarks_bundle
from ltx_trainer.vuga.config import CODE_URL, PAPER_ARXIV, PAPER_URL


def framework_card() -> dict[str, Any]:
    b = benchmarks_bundle()
    jufe = TABLE2_OIQA_VUGA["JUFE-10K"]
    return {
        "name": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "code_url": CODE_URL,
        "problem": (
            "BOIQA models rely on viewport generation (two-step paradigm), creating overhead "
            "and a gap vs mature BIQA. ERP geometry deformation breaks patch-only BIQA models."
        ),
        "method": {
            "paradigm": "Viewport-unaware: raw ERP (or planar image) in, quality score out",
            "backbone": "Frozen SwinV2-T hierarchical features",
            "cmp": "DCN + local-global multiscale perception for ERP deformation",
            "aff": "Top-down adaptive fusion with spatial distortion-aware attention",
            "cae": "Channel-aware enhancement before regression",
        },
        "results": {
            "JUFE-10K_SRCC": jufe["SRCC"],
            "OIQ-10K_SRCC": TABLE2_OIQA_VUGA["OIQ-10K"]["SRCC"],
            "KonIQ_SRCC": b["table3_iqa_vuga"]["KonIQ-10k"]["SRCC"],
        },
        "reference_metrics": b,
        "integration": (
            "GOPEX implements CMP/AFF/CAE stubs, frozen backbone pyramid, MSE training smoke, "
            "and Tables II–VI anchors. Production weights: github.com/KangchengWu/VUGA."
        ),
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.vuga.mock import evaluation_smoke

    return evaluation_smoke()
