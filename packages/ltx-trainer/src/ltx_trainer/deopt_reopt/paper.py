"""Paper cards and knowledge bundle."""

from __future__ import annotations

from typing import Any

from ltx_trainer.deopt_reopt.benchmarks import (
    positioning_vs_direct,
    summary_anchors,
    table_1_kernels,
    table_2_comparison,
)
from ltx_trainer.deopt_reopt.constants import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, UPSTREAM_REPO
from ltx_trainer.deopt_reopt.references import reference_anchors
from ltx_trainer.deopt_reopt.workflows import workflow_card


def paper_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "authors": "Mukunoki, Mikasa, Hayashi, Hoshino, Katagiri",
        "affiliation": "Nagoya University",
        "upstream": UPSTREAM_REPO,
        "findings": [
            "Deopt-Reopt: deoptimize CPU C++ then translate + reoptimize for CUDA",
            "12 HPC kernels, O120 + Q235, Single-shot and Iterative workflows",
            "conv2d clearest win when CPU/GPU designs diverge; not universal",
            "Iterative refinement narrows O120 gap; Q235 retains large D+R advantages",
            "Conditional benefit: performance over successful trials, feasibility varies",
        ],
    }


def knowledge_bundle() -> dict[str, Any]:
    return {
        "paper": paper_card(),
        "workflows": workflow_card(),
        "table_1": table_1_kernels(),
        "table_2_single_shot": table_2_comparison("single_shot"),
        "table_2_iterative": table_2_comparison("iterative"),
        "positioning": positioning_vs_direct(),
        "summary": summary_anchors(),
        "references": reference_anchors(),
    }
