"""Paper cards and knowledge bundle."""

from __future__ import annotations

from typing import Any

from ltx_trainer.midint_division.benchmarks import (
    positioning_vs_cgbn,
    summary_anchors,
    table_1_cgbn_comparison,
)
from ltx_trainer.midint_division.constants import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, UPSTREAM_REPO
from ltx_trainer.midint_division.cost_model import full_mult_bounds
from ltx_trainer.midint_division.references import reference_anchors


def paper_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "authors": "Marchioro, Raahauge, Løvenskjold, Oancea, Watt",
        "affiliations": "DIKU Copenhagen; Waterloo",
        "upstream": UPSTREAM_REPO,
        "findings": [
            "CUDA block-level division via Watt whole shifted inverse (integer Newton)",
            "Precisions 2^13–2^18 bits — beyond cgbn division in experiments",
            "Cost model: 5–7 full classical multiplications; measured ≈5.2× mul at 2^18",
            "Registers + shared staging; prefix scans for compare/subtract",
        ],
    }


def knowledge_bundle() -> dict[str, Any]:
    return {
        "paper": paper_card(),
        "table_1": table_1_cgbn_comparison(),
        "cost_bounds": full_mult_bounds(),
        "positioning": positioning_vs_cgbn(),
        "summary": summary_anchors(),
        "references": reference_anchors(),
    }
