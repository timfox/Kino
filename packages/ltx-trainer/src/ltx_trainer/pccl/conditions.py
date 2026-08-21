"""Condition-based collective pattern representation (Section 4.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class CollectiveCondition:
    chunk_id: int
    src: int
    dests: set[int]
    dist: int = 0


def conditions_all_gather(npus: Iterable[int]) -> list[CollectiveCondition]:
    """All-Gather: each chunk from its source must reach all NPUs in group."""
    group = sorted(npus)
    return [
        CollectiveCondition(chunk_id=i, src=npu, dests=set(group))
        for i, npu in enumerate(group)
    ]


def conditions_all_to_all(npus: Iterable[int]) -> list[CollectiveCondition]:
    """All-to-All: chunk (src, dest) from src to dest for each pair."""
    group = sorted(npus)
    out: list[CollectiveCondition] = []
    cid = 0
    for src in group:
        for dest in group:
            out.append(CollectiveCondition(chunk_id=cid, src=src, dests={dest}))
            cid += 1
    return out


def conditions_all_to_allv(
    npus: Iterable[int],
    *,
    send_counts: dict[int, int] | None = None,
) -> list[CollectiveCondition]:
    """All-to-Allv with per-source send multiplicity."""
    group = sorted(npus)
    counts = send_counts or {n: 1 for n in group}
    out: list[CollectiveCondition] = []
    cid = 0
    for src in group:
        for _ in range(counts.get(src, 1)):
            for dest in group:
                out.append(CollectiveCondition(chunk_id=cid, src=src, dests={dest}))
                cid += 1
    return out


from typing import Callable


def assign_condition_distances(
    conditions: list[CollectiveCondition],
    shortest_path: Callable[[int, int], int],
) -> None:
    """Algorithm 3 lines 1–6: max shortest-path dist per condition."""
    for c in conditions:
        c.dist = max(shortest_path(c.src, d) for d in c.dests)
