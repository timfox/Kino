"""Paper knowledge export."""

from __future__ import annotations

from typing import Any

from ltx_trainer.nvc_erp_qpa.benchmarks import benchmarks_bundle
from ltx_trainer.nvc_erp_qpa.config import (
    LDP_QP_OFFSET_PATTERN,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    Q_NUM,
)
from ltx_trainer.nvc_erp_qpa.datasets import datasets_card
from ltx_trainer.nvc_erp_qpa.pipeline import evaluation_demo_run


def framework_card() -> dict[str, Any]:
    return {
        "title": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "authors": [
            "Daichi Arai",
            "Yuichi Kondo",
            "Kyohei Unno",
            "Yasuko Sugito",
            "Yuichi Kusakabe",
        ],
        "affiliation": "NHK Science & Technology Research Laboratories, Tokyo",
        "components": [
            "Latitude-based adaptive quality q̃φ (extends JVET QPA to NVC)",
            "Vector-bank linear interpolation (DCVC-RT latent modulation)",
            "No retraining — pretrained DCVC-RT + WS-PSNR RD evaluation",
        ],
        "backbone": "DCVC-RT",
        "q_num": Q_NUM,
        "ldp_qp_offset_pattern": list(LDP_QP_OFFSET_PATTERN),
        "contributions": [
            "5.2% BD-Rate WS-PSNR average on JVET class S1 vs DCVC-RT baseline",
            "<0.3% encoding time overhead vs baseline",
            "Stable LDP gains vs VVenC QPA degradation on some sequences",
        ],
        "datasets": datasets_card(),
        "benchmarks": benchmarks_bundle(),
    }


def evaluation_demo() -> dict[str, Any]:
    return {"package": "nvc_erp_qpa", **evaluation_demo_run()}
