"""Paper cards and knowledge bundle."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pccl.benchmarks import (
    parallelism_collectives_table,
    process_group_speedup_anchors,
    scalability_anchors,
    table_i_synthesizer_comparison,
    table_ii_collective_support,
)
from ltx_trainer.pccl.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL
from ltx_trainer.pccl.references import reference_anchors


def paper_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "authors": "William Won, Kartik Lakhotia, Madhu Kumar, Sudarshan Srinivasan, Tushar Krishna",
        "affiliations": "Georgia Tech; Intel Labs; Intel Bengaluru",
        "contributions": [
            "TEN + BFS pathfinding synthesizer",
            "process group-aware routing via full topology",
            "arbitrary collectives including All-to-All / All-to-Allv",
            "512-NPU All-to-All synthesis in 11.68 minutes",
        ],
    }


def knowledge_bundle() -> dict[str, Any]:
    return {
        "paper": paper_card(),
        "table_i": table_i_synthesizer_comparison(),
        "table_ii": table_ii_collective_support(),
        "table_iii_parallelism": parallelism_collectives_table(),
        "scalability": scalability_anchors(),
        "process_group": process_group_speedup_anchors(),
        "references": reference_anchors(),
    }
