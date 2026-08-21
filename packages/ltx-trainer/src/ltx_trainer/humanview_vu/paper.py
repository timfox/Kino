"""Framework card and evaluation demo exports."""

from __future__ import annotations

from typing import Any

from ltx_trainer.humanview_vu.benchmarks import benchmarks_bundle
from ltx_trainer.humanview_vu.config import AWESOME_URL, PAPER_ARXIV, PAPER_TITLE, PAPER_URL
from ltx_trainer.humanview_vu.datasets import datasets_card
from ltx_trainer.humanview_vu.formulation import run_formulation_demo
from ltx_trainer.humanview_vu.future import future_card
from ltx_trainer.humanview_vu.subfields import subfields_card
from ltx_trainer.humanview_vu.taxonomy import taxonomy_card


def framework_card() -> dict[str, Any]:
    return {
        "title": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "awesome": AWESOME_URL,
        "venue_note": "IEEE TPAMI survey submission (arXiv:2606.07433)",
        "core_abilities": ["watch", "remember", "reason"],
        "axes": [
            "unified_formulation_F_VU",
            "watch_remember_reason_taxonomy",
            "subfields_and_benchmarks",
            "training_data_and_eval_landscape",
            "future_directions",
        ],
    }


def evaluation_demo() -> dict[str, Any]:
    return {
        "package": "humanview_vu",
        "framework": framework_card(),
        "taxonomy": taxonomy_card(),
        "formulation": run_formulation_demo(),
        "benchmarks": benchmarks_bundle(),
        "datasets": datasets_card(),
        "subfields": subfields_card(),
        "future": future_card(),
    }


def knowledge_blob() -> dict[str, Any]:
    card = framework_card()
    bench = benchmarks_bundle()
    return {
        "title": card["title"],
        "arxiv": card["arxiv"],
        "url": card["url"],
        "awesome": card["awesome"],
        "summary": (
            "Human-view taxonomy for video MLLMs: watch (perceive evidence), "
            "remember (compact long context), reason (grounded inference). "
            "Unifies VTG, memory, omni-modal, streaming, and o3-like video reasoning."
        ),
        "table1_full_coverage": bench["ours_full_coverage"],
        "eval_dimensions": list(bench["eval_benchmark_dimensions"].keys()),
    }
