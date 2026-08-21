"""Paper cards and knowledge bundle."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fdtd_cpml_multigpu.benchmarks import (
    summary_anchors,
    table_10_single_gpu,
    table_11_strong_scaling_rtx8000,
    table_8_reflection_ratio,
    table_i_positioning,
)
from ltx_trainer.fdtd_cpml_multigpu.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL
from ltx_trainer.fdtd_cpml_multigpu.constants import UPSTREAM_REPO
from ltx_trainer.fdtd_cpml_multigpu.references import reference_anchors


def paper_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "author": "Victory Obieke",
        "affiliation": "Oregon State University",
        "upstream": UPSTREAM_REPO,
        "findings": [
            "direct GPU-to-GPU peer exchange: 2.46–2.76× over host-staged",
            "enlarged ghost regions: modest gain; best at s=4",
            "pencil-yz decomposition wins baseline comparison",
            "multi-GPU chiefly for memory capacity + comm efficiency",
        ],
    }


def knowledge_bundle() -> dict[str, Any]:
    return {
        "paper": paper_card(),
        "table_i": table_i_positioning(),
        "table_8_cpml_reflection": table_8_reflection_ratio(),
        "table_10_single_gpu": table_10_single_gpu(),
        "table_11_strong_scaling": table_11_strong_scaling_rtx8000(),
        "summary": summary_anchors(),
        "references": reference_anchors(),
    }
