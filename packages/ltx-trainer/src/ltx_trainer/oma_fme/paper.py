"""Paper knowledge export."""

from __future__ import annotations

from typing import Any

from ltx_trainer.oma_fme.benchmarks import benchmarks_bundle
from ltx_trainer.oma_fme.config import (
    ENCODER,
    PAPER_ARXIV,
    PAPER_DOI,
    PAPER_TITLE,
    PAPER_URL,
    QPS,
    X265_REUSE_LEVEL,
)
from ltx_trainer.oma_fme.datasets import sjtu_dataset_card
from ltx_trainer.oma_fme.variants import FRAMEWORK_VARIANTS


def paper_knowledge() -> dict[str, Any]:
    return {
        "title": PAPER_TITLE,
        "authors": ["Amritha Premkumar", "Christian Herglotz"],
        "affiliation": "BTU Cottbus-Senftenberg, Chair of Computer Engineering",
        "venue": "MHV '26, Denver",
        "arxiv": PAPER_ARXIV,
        "doi": PAPER_DOI,
        "url": PAPER_URL,
        "license": "CC BY 4.0",
        "contributions": [
            "Projection-aware multirate analysis reuse for 360° ERP and CMP",
            "CRC (HD→4K→8K cascade) vs PRA (per-resolution anchors)",
            "x265 analysis-save/load at reuse level 10",
            "OMAF-compliant DASH packaging",
        ],
        "variants": FRAMEWORK_VARIANTS,
        "encoder": ENCODER,
        "QPs": list(QPS),
        "x265_reuse_level": X265_REUSE_LEVEL,
        "dataset": sjtu_dataset_card(),
        "benchmarks": benchmarks_bundle(),
    }
