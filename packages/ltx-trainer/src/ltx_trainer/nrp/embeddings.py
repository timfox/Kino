"""Embedded database model for NRP (Sec. 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ExtendedArity:
    content_arity: int
    embedding_dim: int

    def __str__(self) -> str:
        return f"{self.content_arity}⟨{self.embedding_dim}⟩"


@dataclass(frozen=True)
class EFact:
    relation: str
    content: tuple[str, ...]
    embedding: tuple[float, ...]

    @property
    def key(self) -> tuple[str, tuple[str, ...]]:
        return self.relation, self.content


class EDatabase:
    """Finite set of e-facts; at most one embedding per (relation, content)."""

    def __init__(self, facts: Iterable[EFact] | None = None) -> None:
        self._facts: dict[tuple[str, tuple[str, ...]], tuple[float, ...]] = {}
        if facts:
            for f in facts:
                self.add(f)

    def add(self, fact: EFact) -> None:
        key = fact.key
        if key in self._facts and self._facts[key] != fact.embedding:
            raise ValueError(f"conflicting embeddings for {key}")
        self._facts[key] = fact.embedding

    def has(self, relation: str, content: tuple[str, ...]) -> bool:
        return (relation, content) in self._facts

    def get(self, relation: str, content: tuple[str, ...]) -> tuple[float, ...] | None:
        return self._facts.get((relation, content))

    def embedding_dim(self, relation: str, content: tuple[str, ...]) -> int:
        emb = self.get(relation, content)
        return len(emb) if emb is not None else 0

    def relation_facts(self, relation: str) -> list[EFact]:
        out: list[EFact] = []
        for (rel, content), emb in self._facts.items():
            if rel == relation:
                out.append(EFact(rel, content, emb))
        return out

    def active_domain(self) -> set[str]:
        dom: set[str] = set()
        for _, content in self._facts:
            dom.update(content)
        return dom

    def merge(self, other: EDatabase) -> EDatabase:
        merged = EDatabase()
        for key, emb in self._facts.items():
            merged._facts[key] = emb
        for key, emb in other._facts.items():
            if key in merged._facts and merged._facts[key] != emb:
                raise ValueError(f"merge conflict on {key}")
            merged._facts[key] = emb
        return merged

    def __iter__(self):
        for (rel, content), emb in sorted(self._facts.items()):
            yield EFact(rel, content, emb)

    def __len__(self) -> int:
        return len(self._facts)


def zero_vector(dim: int) -> tuple[float, ...]:
    return (0.0,) * dim


def sum_aggregate(vectors: Iterable[tuple[float, ...]]) -> tuple[float, ...]:
    vecs = list(vectors)
    if not vecs:
        return ()
    dim = max(len(v) for v in vecs)
    acc = [0.0] * dim
    for v in vecs:
        for i, x in enumerate(v):
            acc[i] += x
    return tuple(acc)


def product_combine(*vectors: tuple[float, ...]) -> tuple[float, ...]:
    """Element-wise product ⊙ with max-dimension padding (Sec. 3)."""
    if not vectors:
        return ()
    dim = max(len(v) for v in vectors)
    out: list[float] = []
    for j in range(dim):
        prod = 1.0
        for v in vectors:
            if len(v) > j:
                prod *= v[j]
        out.append(prod)
    return tuple(out)


def stack_combine(*vectors: tuple[float, ...]) -> tuple[float, ...]:
    """Concatenate (⊕) simulation: take first component of each input in order."""
    return tuple(v[0] if v else 0.0 for v in vectors)
