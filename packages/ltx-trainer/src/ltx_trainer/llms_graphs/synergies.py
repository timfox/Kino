"""Six synergy axes and representative systems (Sec. 4)."""

from __future__ import annotations

from typing import Any

SYNERGY_CATALOG: dict[str, dict[str, Any]] = {
    "llms_for_graphs": {
        "role": "LLMs as predictors/enhancers over graph tasks",
        "topics": ["Graph querying (NLQ→Cypher/GraphQL)", "Graph mining (GraphWiz)", "Graph learning (GNN-LLM alignment)"],
        "anchors": ["GraphWiz", "Neo4j NLQ2Cypher", "GraphTranslator", "One-for-All GNN"],
    },
    "graphs_for_llms": {
        "role": "Graph RAG supplies relationship-rich context",
        "topics": ["Community summaries", "Attributed subgraph retrieval", "Graph-indexed RAG"],
        "anchors": ["GraphRAG", "ArchRAG", "GRAG", "PathRAG"],
    },
    "kgs_for_llms": {
        "role": "KGs ground, guide, and validate LLM reasoning",
        "topics": ["Background knowledge", "Reasoning guidelines", "Refiners & validators"],
        "anchors": ["KG-RAG", "ToG", "GCR", "KG-Agent", "KGR"],
    },
    "llms_for_kgs": {
        "role": "LLMs construct, complete, and query KGs",
        "topics": ["KG creation", "Completion", "Embedding", "Text2SPARQL", "Analytics"],
        "anchors": ["KEPLER", "K-BERT", "UrbanKGent", "Text-to-SPARQL"],
    },
    "graphs_for_agents": {
        "role": "Graph structures for planning, tools, memory, coordination",
        "topics": ["Task planning graphs", "Tool-calling graphs", "Memory graphs", "Multi-agent graphs"],
        "anchors": ["Graph-of-Thought", "ToolNet", "HippoRAG", "DynTaskMAS"],
    },
    "agents_for_graphs": {
        "role": "Agentic decomposition + graph tool invocation",
        "topics": ["Graph reasoning", "Text2Cypher repair", "KG construction agents"],
        "anchors": ["GDS Agent", "Multi-agent GraphRAG", "MA-GTS", "Scalable graph reasoning agents"],
    },
}


def synergy_summary() -> list[dict[str, Any]]:
    return [{"axis": k, **v} for k, v in SYNERGY_CATALOG.items()]


def count_anchors() -> int:
    return sum(len(v["anchors"]) for v in SYNERGY_CATALOG.values())


def synergies_demo() -> dict[str, Any]:
    return {
        "axes": len(SYNERGY_CATALOG),
        "anchor_systems": count_anchors(),
        "bidirectional": True,
    }
