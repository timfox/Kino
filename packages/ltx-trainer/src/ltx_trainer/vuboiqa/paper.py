"""Framework summary for agents."""

from __future__ import annotations

from typing import Any

from ltx_trainer.vuboiqa.config import GITHUB_URL, PAPER_ARXIV, PAPER_DOI, PAPER_TITLE, PAPER_URL


def framework_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "doi": PAPER_DOI,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "code": GITHUB_URL,
        "components": [
            "APS: prior-equator ERP patch sampling (viewport-unaware)",
            "PDFF: deformable multi-scale fusion + DAA (Swin-V2 backbone stub)",
            "LGQA: HPA + patch self-attention → global MOS",
            "norm-in-norm training loss; adapts to 2D-IQA",
        ],
        "outputs": ["perceptual_quality_score"],
    }
