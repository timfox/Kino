"""Interval predicates for IFANN / ISANN / RFANN / RSANN."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class QueryType(str, Enum):
    IFANN = "IF"
    ISANN = "IS"
    RFANN = "RF"  # scalar in range
    RSANN = "RS"  # timestamp stabbing


@dataclass(frozen=True)
class Interval:
    left: float
    right: float

    def __post_init__(self) -> None:
        if self.left > self.right:
            raise ValueError(f"invalid interval [{self.left}, {self.right}]")

    @property
    def mid(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def length(self) -> float:
        return self.right - self.left

    def union(self, other: Interval) -> Interval:
        return Interval(min(self.left, other.left), max(self.right, other.right))

    def intersect(self, other: Interval) -> Interval | None:
        lo = max(self.left, other.left)
        hi = min(self.right, other.right)
        if lo > hi:
            return None
        return Interval(lo, hi)

    def contains(self, other: Interval) -> bool:
        return self.left <= other.left and other.right <= self.right

    def subset_of(self, other: Interval) -> bool:
        return other.contains(self)


FLAG_IF = 1
FLAG_IS = 2


def phi_if(i_u: Interval, i_v: Interval, i_w: Interval) -> bool:
    """Φ_IF: I_w ⊆ I_u ∪ I_v."""
    return i_w.subset_of(i_u.union(i_v))


def phi_is(i_u: Interval, i_v: Interval, i_w: Interval) -> bool:
    """Φ_IS: I_u ∩ I_v ⊆ I_w (only when intersection non-empty)."""
    inter = i_u.intersect(i_v)
    if inter is None:
        return False
    return inter.subset_of(i_w)


def node_valid(obj_interval: Interval, query_interval: Interval, qtype: QueryType) -> bool:
    if qtype in (QueryType.IFANN, QueryType.RFANN):
        return obj_interval.subset_of(query_interval)
    if qtype in (QueryType.ISANN, QueryType.RSANN):
        return query_interval.subset_of(obj_interval)
    raise ValueError(qtype)


def edge_active(mask: int, qtype: QueryType) -> bool:
    if qtype in (QueryType.IFANN, QueryType.RFANN):
        return bool(mask & FLAG_IF)
    return bool(mask & FLAG_IS)


def brute_force_knn(
    vectors: dict[str, tuple[float, ...]],
    query: tuple[float, ...],
    k: int,
    *,
    valid_ids: Iterable[str] | None = None,
) -> list[str]:
    ids = list(valid_ids) if valid_ids is not None else list(vectors.keys())

    def dist(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5

    ranked = sorted(ids, key=lambda i: dist(vectors[i], query))
    return ranked[:k]
