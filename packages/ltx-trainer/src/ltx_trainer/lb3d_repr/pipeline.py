"""Framework card, tables, and demo entry points."""

from __future__ import annotations

from typing import Any

from ltx_trainer.lb3d_repr.benchmarks import (
    paper_anchors,
    table_3dgs_optimization,
    table_explicit_representations,
)
from ltx_trainer.lb3d_repr.config import (
    LB3DReprConfig,
    PAPER_ARXIV,
    PAPER_AUTHORS,
    PAPER_TITLE,
    PAPER_URL,
    PAPER_VENUE,
)
from ltx_trainer.lb3d_repr.integration import (
    gopex_stub_links,
    ltx_nvs_pipeline_notes,
    survey_to_gopex_mapping,
)
from ltx_trainer.lb3d_repr.taxonomy import (
    UNIFIED_FORMULATION,
    acceleration_strategies,
    future_directions,
    landscape_nodes,
    survey_sections,
    taxonomy_tree,
)
from ltx_trainer.lb3d_repr.tradeoffs import representation_tradeoff_report


def framework_card(cfg: LB3DReprConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LB3DReprConfig()
    return {
        "paper": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "authors": PAPER_AUTHORS,
        "venue": PAPER_VENUE,
        "url": PAPER_URL,
        "formulation": UNIFIED_FORMULATION,
        "sections": survey_sections(),
        "taxonomy": taxonomy_tree(),
        "landscape_nodes": [n["name"] for n in landscape_nodes()],
        "acceleration_strategies": [s["id"] for s in acceleration_strategies()],
        "future_directions": future_directions(),
        "gopex_stubs": list(gopex_stub_links().keys()),
        "config": cfg.__dict__,
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_explicit": table_explicit_representations(),
        "table2_3dgs_optimization": table_3dgs_optimization(),
        "paper_anchors": paper_anchors(),
        "taxonomy": taxonomy_tree(),
        "acceleration": acceleration_strategies(),
        "framework": framework_card(),
    }


def evaluation_demo(*, seed: int = 0) -> dict[str, Any]:
    return {
        "tradeoffs": representation_tradeoff_report(seed=seed),
        "gopex_mapping": survey_to_gopex_mapping(),
        "ltx_notes": ltx_nvs_pipeline_notes(),
        "mesh_row": next(
            (r for r in table_explicit_representations() if "Mesh" in r["representation"]),
            None,
        ),
    }


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    demo = evaluation_demo(seed=seed)
    trade = demo["tradeoffs"]
    return {
        "paper": "lb3d_repr",
        "arxiv": PAPER_ARXIV,
        "explicit_rows": len(table_explicit_representations()),
        "gs_opt_rows": len(table_3dgs_optimization()),
        "gopex_stub_count": len(gopex_stub_links()),
        "best_repr": trade["best_for_deployability"],
        "best_tradeoff_index": trade["best_index"],
        "hybrid_repr_count": trade["hybrid_count"],
    }
