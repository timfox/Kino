"""Index construction: DBA and MBA (Sec. V)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.kd_truss.mts import delta_triangle_list
from ltx_trainer.kd_truss.tc_index import TcIndex
from ltx_trainer.kd_truss.truss import online_kd_truss_query


def dba_build_tc_index(edges: set[tuple[int, int]], k_max: int, delta_max: int) -> dict[str, Any]:
    """Decomposition Based Algorithm: H-IES via horizontal decomph (Sec. V-A)."""
    _, buckets = delta_triangle_list(edges)
    h_ies_count = 0
    for k in range(3, k_max + 1):
        prev = online_kd_truss_query(edges, k, delta_max)
        for delta in range(delta_max - 1, -1, -1):
            cur = online_kd_truss_query(edges, k, delta)
            h_ies_count += len(prev - cur)
            prev = cur
    idx = TcIndex.build(edges, k_max=k_max)
    return {"algorithm": "DBA", "h_ies_total": h_ies_count, "k_maps": len(idx.maps), "delta_buckets": len(buckets)}


def mba_build_indexes(edges: set[tuple[int, int]], k_max: int, delta_max: int) -> dict[str, Any]:
    """Maintenance Based Algorithm: V-IES + H-IES in one δ sweep (Sec. V-B)."""
    tri_mts, buckets = delta_triangle_list(edges)
    invalidated = sum(len(b) for b in buckets)
    tc = TcIndex.build(edges, k_max=k_max)
    from ltx_trainer.kd_truss.dc_index import DcIndex

    dc = DcIndex.build(edges, k_max=k_max, delta_max=delta_max)
    return {
        "algorithm": "MBA",
        "triangles_invalidated_once": invalidated,
        "tc_maps": len(tc.maps),
        "dc_lookup": len(dc.lookup),
    }


def construction_demo() -> dict[str, Any]:
    from ltx_trainer.kd_truss.running_example import FIG1_TIMESTAMPS

    edges = {tuple(e) for e in FIG1_TIMESTAMPS}
    dba = dba_build_tc_index(edges, k_max=5, delta_max=6)
    mba = mba_build_indexes(edges, k_max=5, delta_max=6)
    return {"dba": dba, "mba": mba}
