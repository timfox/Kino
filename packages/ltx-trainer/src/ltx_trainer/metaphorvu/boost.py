"""MetaphorBoost inference-time mapping augmentation."""

from __future__ import annotations

from typing import Any

from ltx_trainer.metaphorvu.config import MetaphorVUConfig
from ltx_trainer.metaphorvu.kg import (
    MetaphorKnowledgeGraph,
    identify_keywords_stub,
    top_z_references,
)


def build_demo_graph() -> MetaphorKnowledgeGraph:
    """Mini graph mirroring tailcoat-pig example in Figure 6."""
    pairs = [
        ("pig", "greedy"),
        ("tailcoat", "nobility"),
        ("nobility", "wealth"),
        ("wealth", "power"),
        ("judge", "authority"),
        ("banquet", "waste"),
        ("cat", "underprivileged"),
        ("scrap", "poverty"),
        ("ruling group", "plunder"),
        ("greedy", "plunder"),
        ("power", "ruling group"),
    ]
    return MetaphorKnowledgeGraph.from_pairs(pairs)


def metaphor_boost(
    visual_elements: list[str],
    graph: MetaphorKnowledgeGraph | None = None,
    cfg: MetaphorVUConfig | None = None,
) -> dict[str, Any]:
    """
    MetaphorBoost pipeline: identify → KG query → references for generation (Eq. 4–6).

    Returns keywords ``K`` and reference concepts ``R``; generation is out of scope here.
    """
    cfg = cfg or MetaphorVUConfig()
    graph = graph or build_demo_graph()
    keywords = identify_keywords_stub(visual_elements)
    references = top_z_references(
        keywords,
        graph,
        hops=cfg.query_hops,
        z=cfg.top_z,
    )
    return {
        "keywords": keywords,
        "references": references,
        "num_keywords": len(keywords),
        "num_references": len(references),
        "hops": cfg.query_hops,
        "top_z": cfg.top_z,
    }
