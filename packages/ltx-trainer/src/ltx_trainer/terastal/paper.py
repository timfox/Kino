"""Paper cards and knowledge bundle."""

from __future__ import annotations

from typing import Any

from ltx_trainer.terastal.benchmarks import (
    miss_rate_comparison_anchors,
    related_work_positioning,
    summary_anchors,
    table_i_hardware,
    table_ii_workloads,
)
from ltx_trainer.terastal.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL
from ltx_trainer.terastal.constants import PAPER_VENUE
from ltx_trainer.terastal.references import reference_anchors


def paper_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "venue": PAPER_VENUE,
        "authors": ["Sing-Yao Wu", "Fengshuo Song", "Eli Bozorgzadeh"],
        "affiliation": "University of California, Irvine",
        "findings": [
            "layer variants (S2D/D2S) shrink cross-accelerator latency gaps",
            "virtual budgets (Algorithm 1) guide offline variant selection",
            "slack-aware scheduler (Algorithm 2) maps layers + variants online",
            "40.58% / 30.53% / 36.27% miss-rate reduction vs FCFS / EDF / DREAM",
            "2.24% average normalized accuracy loss with variants",
        ],
    }


def knowledge_bundle() -> dict[str, Any]:
    return {
        "paper": paper_card(),
        "table_i": table_i_hardware(),
        "table_ii": table_ii_workloads(),
        "positioning": related_work_positioning(),
        "miss_rate_anchors": miss_rate_comparison_anchors(),
        "summary": summary_anchors(),
        "references": reference_anchors(),
    }
