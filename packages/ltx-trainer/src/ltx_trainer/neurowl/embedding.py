"""Embedding rankers — OnT / SBERT stand-ins (CPU)."""

from __future__ import annotations

from ltx_trainer.neurowl.ontology import ELOntology


def _token_overlap(a: str, b: str) -> float:
    ta = set(a.lower().replace("_", " ").split())
    tb = set(b.lower().replace("_", " ").split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def subsumption_score(
    ontology: ELOntology,
    child: str,
    parent: str,
    *,
    backend: str = "ont",
) -> float:
    """Higher = more likely child ⊑ parent (proxy for OnT/SBERT)."""
    la = ontology.labels.get(child, child)
    lb = ontology.labels.get(parent, parent)
    base = _token_overlap(la, lb)
    # Prefer shorter taxonomic distance when both in ontology
    bonus = 0.0
    if parent in ontology.concepts() and child in ontology.concepts():
        # mild prior: shared ancestors / known edges
        if parent in ontology.parents(child):
            bonus = 0.5
        elif child in ontology.children(parent):
            bonus = 0.35
    # OnT fine-tune proxy: slightly stronger structural prior
    if backend == "ont":
        bonus *= 1.15
        base = 0.55 * base + 0.45 * (base + bonus)
    else:
        base = 0.7 * base + 0.3 * (base + bonus)
    return min(1.0, base + bonus)


def bridge_score(ontology: ELOntology, a: str, c: str, b: str, *, backend: str = "ont") -> float:
    """¯s(C) = ½ (s(A⊑C) + s(C⊑B))."""
    return 0.5 * (
        subsumption_score(ontology, a, c, backend=backend)
        + subsumption_score(ontology, c, b, backend=backend)
    )


def rank_candidates(
    ontology: ELOntology,
    pairs: list[tuple[str, str]],
    *,
    backend: str = "ont",
    top_k: int = 10,
) -> list[tuple[str, str, float]]:
    scored = [
        (c, p, subsumption_score(ontology, c, p, backend=backend)) for c, p in pairs
    ]
    scored.sort(key=lambda t: t[2], reverse=True)
    return scored[:top_k]
