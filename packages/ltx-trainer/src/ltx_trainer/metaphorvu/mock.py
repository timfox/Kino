"""MetaphorVU KG query smoke (arXiv:2605.25461)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.metaphorvu.config import MetaphorVUConfig
from ltx_trainer.metaphorvu.kg import MetaphorKnowledgeGraph


def evaluation_smoke(cfg: MetaphorVUConfig | None = None) -> dict[str, Any]:
    c = cfg or MetaphorVUConfig()
    g = MetaphorKnowledgeGraph.from_pairs([("time", "river"), ("river", "flow")])
    nbr = g.neighbors_within_hops("time", hops=2)
    return {
        "paper": c.paper_arxiv,
        "reachable_nodes": len(nbr),
    }
