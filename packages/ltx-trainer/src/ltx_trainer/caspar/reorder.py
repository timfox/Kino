"""Adaptive topological reordering to reduce register pressure (§III)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.caspar.dabseg import DabsegGraph, NodeKind


@dataclass(frozen=True)
class ReorderKey:
    """Lexicographic sort key (R0,A0,R1,A1,R2,A2,R3,A3,K) from paper §III-A."""

    r0: int
    a0: int
    r1: int
    a1: int
    r2: int
    a2: int
    r3: int
    a3: int
    k: int

    def as_tuple(self) -> tuple[int, ...]:
        return (self.r0, self.a0, self.r1, self.a1, self.r2, self.a2, self.r3, self.a3, self.k)


def _live_values(graph: DabsegGraph, scheduled: set[int], produced: set[int]) -> set[int]:
    live: set[int] = set()
    for vid in produced:
        still_needed = False
        for call_id in graph.dependents(vid):
            if call_id not in scheduled:
                still_needed = True
                break
        if still_needed:
            live.add(vid)
    return live


def _release_potential(graph: DabsegGraph, call_id: int, live: set[int], scheduled: set[int]) -> tuple[int, int, int, int]:
    """Approximate R0..R3: values freed now or within i subsequent calls."""
    call = graph.nodes[call_id]
    freed_now = sum(1 for v in call.inputs if v in live and not graph.dependents(v))
    # Stub: higher-order release counts inputs consumed by near-future calls.
    r1 = sum(1 for v in call.inputs if v in live) - freed_now
    r2 = max(0, len(call.inputs) // 2 - freed_now)
    r3 = max(0, len(call.inputs) - freed_now - r1)
    return freed_now, r1, r2, r3


def _affinity(graph: DabsegGraph, call_id: int, live: set[int]) -> tuple[int, int, int, int]:
    """A0..A3: co-use affinity with live values (stub heuristic)."""
    call = graph.nodes[call_id]
    shared = sum(1 for v in call.inputs if v in live)
    return shared, max(0, shared - 1), max(0, shared // 2), 0


def reorder_calls(graph: DabsegGraph) -> list[int]:
    """
    Greedy topological order of call nodes using paper sort key.
    Returns ordered call node ids.
    """
    calls = graph.call_nodes()
    call_ids = [c.id for c in calls]
    # Deterministic depth-first preorder key K.
    k_map = {cid: i for i, cid in enumerate(call_ids)}

    inputs_ready: set[int] = {v.id for v in graph.value_nodes() if v.meta.get("symbol")}

    scheduled: set[int] = set()
    produced: set[int] = set(v.id for v in graph.value_nodes() if v.meta.get("symbol"))
    order: list[int] = []

    while len(scheduled) < len(call_ids):
        live = _live_values(graph, scheduled, produced)
        firable: list[int] = []
        for cid in call_ids:
            if cid in scheduled:
                continue
            call = graph.nodes[cid]
            if all(inp in produced or inp in inputs_ready for inp in call.inputs):
                firable.append(cid)
        if not firable:
            # Fallback: schedule remaining in K order.
            firable = [cid for cid in call_ids if cid not in scheduled]

        keys: list[tuple[tuple[int, ...], int]] = []
        for cid in firable:
            r0, r1, r2, r3 = _release_potential(graph, cid, live, scheduled)
            a0, a1, a2, a3 = _affinity(graph, cid, live)
            key = ReorderKey(r0, a0, r1, a1, r2, a2, r3, a3, k_map[cid])
            keys.append((key.as_tuple(), cid))
        keys.sort()
        pick = keys[0][1]
        scheduled.add(pick)
        order.append(pick)
        for v in graph.nodes[pick].outputs:
            produced.add(v)

    return order


def estimate_register_pressure(order: list[int], graph: DabsegGraph) -> int:
    """Peak live value count under a schedule (register pressure proxy)."""
    scheduled: set[int] = set()
    produced: set[int] = {v.id for v in graph.value_nodes() if v.meta.get("symbol")}
    peak = 0
    for cid in order:
        scheduled.add(cid)
        for v in graph.nodes[cid].outputs:
            produced.add(v)
        live = _live_values(graph, scheduled, produced)
        peak = max(peak, len(live))
    return peak
