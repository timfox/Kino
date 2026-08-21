"""Formal PDE definitions (Sec. 3)."""

from __future__ import annotations

from collections.abc import Callable, Iterable


def match_instances(x: str, corpus: Iterable[str], *, matcher: Callable[[str, str], bool] | None = None) -> bool:
    """b(x, x') — default exact string match."""
    fn = matcher or (lambda a, b: a == b)
    return any(fn(x, c) for c in corpus)


def instance_exposed(model_id: str, x: str, pretrain_corpus: Iterable[str]) -> int:
    """f(M, x) ∈ {0,1} (Eq. 1)."""
    _ = model_id
    return 1 if match_instances(x, pretrain_corpus) else 0


def dataset_partially_exposed(model_id: str, dataset: Iterable[str], pretrain_corpus: Iterable[str]) -> bool:
    """∃x ∈ D, f(M,x)=1 (Eq. 2)."""
    return any(instance_exposed(model_id, x, pretrain_corpus) for x in dataset)


def dataset_fully_exposed(model_id: str, dataset: Iterable[str], pretrain_corpus: Iterable[str]) -> bool:
    """∀x ∈ D, f(M,x)=1 (Eq. 3)."""
    items = list(dataset)
    if not items:
        return False
    return all(instance_exposed(model_id, x, pretrain_corpus) for x in items)


def exposure_score(model_id: str, dataset: Iterable[str], pretrain_corpus: Iterable[str]) -> float:
    """PDE(D,M) = sum f(M,x) / |D| (Eq. 4)."""
    items = list(dataset)
    if not items:
        return 0.0
    exposed = sum(instance_exposed(model_id, x, pretrain_corpus) for x in items)
    return exposed / len(items)
