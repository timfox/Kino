"""Tutorial tables and anchors (Khan et al., arXiv:2606.11560)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.llms_graphs.config import (
    PAPER_ARXIV,
    PAPER_AUTHORS,
    PAPER_REPO,
    PAPER_TITLE,
    PAPER_URL,
    PAPER_VENUE,
    TUTORIAL_SECTIONS,
)
from ltx_trainer.llms_graphs.synergies import SYNERGY_CATALOG, count_anchors

# Table: Graph RAG methods (Sec. 4.2, Zhou et al. VLDM survey)
TABLE_GRAPH_RAG: list[dict[str, Any]] = [
    {"method": "GraphRAG", "mechanism": "Document graph + Leiden communities + summaries", "ref": "Edge et al. 2024"},
    {"method": "ArchRAG", "mechanism": "Attributed community hierarchical retrieval", "ref": "Wang et al. AAAI 2026"},
    {"method": "GRAG", "mechanism": "Top-k subgraph retrieval + embedding alignment", "ref": "Hu et al. 2024"},
    {"method": "PathRAG", "mechanism": "Relational path pruning for RAG", "ref": "Chen et al. 2025"},
]

# Table: KG roles for LLMs (Sec. 4.3)
TABLE_KG_ROLES: list[dict[str, str]] = [
    {"category": "Background Knowledge", "representatives": "GRAG, KG-Adapter, InfuserKI, KG-RAG"},
    {"category": "Reasoning Guidelines", "representatives": "ToG, GCR, KG-Agent, ODA"},
    {"category": "Refiners & Validators", "representatives": "KGR, KG-Rank, EFSUM, LPKG"},
]

# Table: LLMs for KG tasks (Sec. 4.4)
TABLE_LLM_FOR_KG: list[dict[str, str]] = [
    {"task": "KG Creation", "representatives": "UrbanKGent, multi-modal extraction"},
    {"task": "KG Completion", "representatives": "Seq2seq triple generation, KG-BERT"},
    {"task": "KG Embedding", "representatives": "KEPLER, K-BERT"},
    {"task": "KG Querying", "representatives": "Text2SPARQL, NLQ grounding"},
    {"task": "KG Analytics", "representatives": "Language is all a graph needs"},
]

# Future directions (Sec. 4.7)
FUTURE_DIRECTIONS: list[str] = [
    "Unification of LLM + KG + Vector DB & NeuroSymbolic AI",
    "KG-based agentic memory",
    "Unifying long context with RAG",
    "Explainability via graph-grounded evidence traces",
    "Security and privacy for KG/agent pipelines",
]

PAPER_ANCHORS = {
    "synergy_axes": len(SYNERGY_CATALOG),
    "anchor_systems": count_anchors(),
    "tutorial_minutes": sum(m for _, m in TUTORIAL_SECTIONS),
    "tutorial_sections": len(TUTORIAL_SECTIONS),
    "graph_rag_methods": len(TABLE_GRAPH_RAG),
    "future_directions": len(FUTURE_DIRECTIONS),
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": {
            "arxiv": PAPER_ARXIV,
            "title": PAPER_TITLE,
            "authors": PAPER_AUTHORS,
            "venue": PAPER_VENUE,
            "url": PAPER_URL,
            "repo": PAPER_REPO,
        },
        "tutorial_outline": [{"section": s, "minutes": m} for s, m in TUTORIAL_SECTIONS],
        "synergy_catalog": {k: v for k, v in SYNERGY_CATALOG.items()},
        "table_graph_rag": TABLE_GRAPH_RAG,
        "table_kg_roles": TABLE_KG_ROLES,
        "table_llm_for_kg": TABLE_LLM_FOR_KG,
        "future_directions": FUTURE_DIRECTIONS,
        "anchors": PAPER_ANCHORS,
    }
