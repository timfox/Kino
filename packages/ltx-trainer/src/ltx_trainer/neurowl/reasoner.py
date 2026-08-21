"""Stage 1: logical checking via transitive closure (atomic EL)."""

from __future__ import annotations

from ltx_trainer.neurowl.ontology import ELOntology, Subsumption


def entails(ontology: ELOntology, child: str, parent: str) -> bool:
    """Return True iff ontology |= child ⊑ parent (reflexive-transitive)."""
    if child == parent:
        return True
    adj = ontology.adjacency()
    seen: set[str] = set()
    stack = [child]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nxt in adj.get(cur, ()):
            if nxt == parent:
                return True
            stack.append(nxt)
    return False


def justification(ontology: ELOntology, child: str, parent: str) -> list[Subsumption]:
    """Minimal path of asserted axioms witnessing child ⊑ parent (BFS parents)."""
    if child == parent:
        return []
    if not entails(ontology, child, parent):
        return []
    # Reconstruct one path of edges
    adj = ontology.adjacency()
    prev: dict[str, tuple[str, Subsumption] | None] = {child: None}
    queue = [child]
    found = False
    while queue and not found:
        cur = queue.pop(0)
        for nxt in adj.get(cur, ()):
            if nxt in prev:
                continue
            edge = Subsumption(cur, nxt)
            prev[nxt] = (cur, edge)
            if nxt == parent:
                found = True
                break
            queue.append(nxt)
    if not found:
        return []
    path: list[Subsumption] = []
    node = parent
    while node != child:
        entry = prev[node]
        assert entry is not None
        _prev_node, edge = entry
        path.append(edge)
        node = _prev_node
    path.reverse()
    return path


def direct_children(ontology: ELOntology, concept: str) -> list[str]:
    return sorted(ontology.children(concept))


def direct_parents(ontology: ELOntology, concept: str) -> list[str]:
    return sorted(ontology.parents(concept))
