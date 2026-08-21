"""Temporal Containment Index — TC-Index (Sec. IV-A)."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.kd_truss.truss import edge_k_span, online_kd_truss_query, static_k_truss


@dataclass
class TcMap:
    """Ik = (Ek, Dk) per Def. 5 / Sec. IV-A."""

    k: int
    ek: list[tuple[int, int]] = field(default_factory=list)
    dk: dict[int, int] = field(default_factory=dict)  # k-span -> start offset in ek


@dataclass
class TcIndex:
    maps: dict[int, TcMap] = field(default_factory=dict)

    @classmethod
    def build(cls, edges: set[tuple[int, int]], k_max: int | None = None) -> TcIndex:
        idx = cls()
        if k_max is None:
            k_max = max(3, _estimate_kmax(edges))
        for k in range(3, k_max + 1):
            k_truss = static_k_truss(edges, k)
            if not k_truss:
                continue
            spans = [(edge_k_span(edges, k, e), e) for e in k_truss]
            spans.sort(key=lambda x: (-x[0], x[1]))
            tm = TcMap(k=k)
            tm.ek = [e for _, e in spans]
            offsets: dict[int, int] = {}
            for i, (sp, _) in enumerate(spans):
                if sp not in offsets:
                    offsets[sp] = i
            tm.dk = offsets
            idx.maps[k] = tm
        return idx

    def tc_query(self, k: int, delta: int) -> set[tuple[int, int]]:
        """TC-Query: return edges with k-span ≤ δ (Theorem 2)."""
        if k not in self.maps:
            return set()
        tm = self.maps[k]
        eligible = sorted(s for s in tm.dk if s <= delta)
        if not eligible:
            return set()
        start = tm.dk[eligible[-1]]
        return set(tm.ek[start:])


def _estimate_kmax(edges: set[tuple[int, int]]) -> int:
    best = 2
    for k in range(3, 20):
        if static_k_truss(edges, k):
            best = k
        else:
            break
    return best


def tc_index_demo() -> dict[str, Any]:
    from ltx_trainer.kd_truss.running_example import FIG1_TIMESTAMPS

    edges = {tuple(e) for e in FIG1_TIMESTAMPS}
    idx = TcIndex.build(edges, k_max=5)
    online = online_kd_truss_query(edges, k=4, delta=1)
    indexed = idx.tc_query(k=4, delta=1)
    return {
        "k_maps": len(idx.maps),
        "tc_matches_online": online == indexed,
        "k4_delta1_count": len(indexed),
    }
