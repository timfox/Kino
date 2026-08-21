"""KG grounding roles: background, reasoning, validation (Sec. 4.3)."""

from __future__ import annotations

from typing import Any

KG_ROLES: list[dict[str, str]] = [
    {"role": "Background Knowledge", "mechanism": "Subgraph-text alignment, adapter injection", "examples": "InfuserKI, KG-Adapter, GRAG"},
    {"role": "Reasoning Guidelines", "mechanism": "KG-guided decoding and sequential traversal", "examples": "ToG, GCR, LLM-ARK, KG-Agent"},
    {"role": "Refiners & Validators", "mechanism": "Re-rank and verify against KG facts", "examples": "KGR, KG-Rank, EFSUM, Interactive-KBQA"},
]


def kg_triple_filter(
    triples: list[tuple[str, str, str]],
    query_entities: set[str],
    *,
    max_hops: int = 2,
) -> list[tuple[str, str, str]]:
    """Retrieve top-k relevant triples by entity overlap (KG-RAG stub)."""
    scored = []
    for h, r, t in triples:
        ents = {h.lower(), t.lower()}
        overlap = len(ents & {e.lower() for e in query_entities})
        if overlap > 0:
            scored.append((overlap, (h, r, t)))
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[: max_hops * 5]]


def validate_answer(answer: str, triples: list[tuple[str, str, str]]) -> dict[str, Any]:
    """KGR-style factual consistency check (lexical overlap stub)."""
    answer_tokens = set(answer.lower().split())
    supported = 0
    for h, r, t in triples:
        fact_tokens = set(f"{h} {r} {t}".lower().split())
        if answer_tokens & fact_tokens:
            supported += 1
    return {
        "supported_fact_hits": supported,
        "grounded": supported > 0,
    }


def kg_grounding_demo() -> dict[str, Any]:
    triples = [
        ("Paris", "capital_of", "France"),
        ("France", "in", "Europe"),
        ("Berlin", "capital_of", "Germany"),
    ]
    retrieved = kg_triple_filter(triples, {"Paris", "France"})
    val = validate_answer("Paris is the capital of France", retrieved)
    return {"roles": len(KG_ROLES), "triples_retrieved": len(retrieved), "grounded": val["grounded"]}
