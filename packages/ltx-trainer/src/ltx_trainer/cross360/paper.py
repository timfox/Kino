"""Paper knowledge export."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cross360.benchmarks import benchmarks_bundle
from ltx_trainer.cross360.config import CODE_URL, PAPER_ARXIV, PAPER_TITLE, PAPER_URL
from ltx_trainer.cross360.datasets import all_datasets_card
from ltx_trainer.cross360.pipeline import evaluation_demo_run


def framework_card() -> dict[str, Any]:
    return {
        "title": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "code": CODE_URL,
        "authors": ["Kun Huang", "Fang-Lue Zhang", "Neil Dodgson"],
        "affiliation": "Victoria University of Wellington",
        "modules": ["CPFA", "PFAA", "ERP encoder (ResNet34 + fine conv)"],
        "projections": ["ERP (global)", "TP tangent patches (local, N=26)"],
        "contributions": [
            "Cross-attention aligns TP patches with full-FoV ERP at each decoder scale",
            "PFAA progressively aggregates multi-scale decoded features",
            "SOTA on M3D and Structured3D among fair-comparison methods",
        ],
        "datasets": all_datasets_card(),
        "benchmarks": benchmarks_bundle(),
    }


def evaluation_demo() -> dict[str, Any]:
    return {"package": "cross360", **evaluation_demo_run()}
