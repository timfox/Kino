"""Graph RAG stub: community retrieval + subgraph context (Sec. 4.2)."""

from __future__ import annotations

from typing import Any

import numpy as np


def leiden_communities(adjacency: np.ndarray, seed: int = 0) -> list[list[int]]:
    """Toy community partition via label propagation (stub for GraphRAG/ArchRAG)."""
    n = adjacency.shape[0]
    rng = np.random.default_rng(seed)
    labels = np.arange(n)
    for _ in range(5):
        order = rng.permutation(n)
        for v in order:
            nbrs = np.where(adjacency[v] > 0)[0]
            if len(nbrs) == 0:
                continue
            counts: dict[int, int] = {}
            for u in nbrs:
                counts[int(labels[u])] = counts.get(int(labels[u]), 0) + 1
            labels[v] = max(counts, key=counts.get)
    communities: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        communities.setdefault(int(lab), []).append(i)
    return list(communities.values())


def community_summary(community: list[int], node_text: dict[int, str]) -> str:
    texts = [node_text.get(i, f"node_{i}") for i in community]
    return "; ".join(texts[:5]) + ("..." if len(texts) > 5 else "")


def graph_rag_retrieve(
    query: str,
    adjacency: np.ndarray,
    node_text: dict[int, str],
    *,
    top_k_communities: int = 2,
    seed: int = 0,
) -> dict[str, Any]:
    """Retrieve community summaries ranked by lexical overlap (ArchRAG-style stub)."""
    communities = leiden_communities(adjacency, seed=seed)
    q_tokens = set(query.lower().split())
    scored = []
    for comm in communities:
        summary = community_summary(comm, node_text)
        overlap = len(q_tokens & set(summary.lower().split()))
        scored.append((overlap, comm, summary))
    scored.sort(key=lambda x: -x[0])
    top = scored[:top_k_communities]
    return {
        "query": query,
        "communities_retrieved": len(top),
        "context": [s for _, _, s in top],
        "total_communities": len(communities),
    }


def graph_rag_demo(*, seed: int = 0) -> dict[str, Any]:
    adj = np.array(
        [
            [0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
        ],
        dtype=np.float64,
    )
    text = {0: "Alice works at GraphLab", 1: "Bob collaborates with Alice", 2: "GraphLab builds RAG", 3: "KG stores triples", 4: "RAG uses communities"}
    out = graph_rag_retrieve("Graph RAG communities", adj, text, seed=seed)
    return {"retrieved": out["communities_retrieved"], "has_context": len(out["context"]) > 0}
