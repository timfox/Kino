"""Framework card, evaluation demo, and smoke entry points."""

from __future__ import annotations

from typing import Any

from ltx_trainer.urng.benchmarks import PAPER_ANCHORS, benchmarks_bundle
from ltx_trainer.urng.config import UrngConfig
from ltx_trainer.urng.integration import integration_bundle
from ltx_trainer.urng.prune import build_ug, edge_mask
from ltx_trainer.urng.query import (
    brute_force_query,
    build_entry_arrays,
    get_entry_node,
    interval_aware_beam_search,
)
from ltx_trainer.urng.running_example import EXAMPLE1_BITS, FIG2_NODES, fig2_dataset


def _fig2_bundle() -> tuple[list[str], dict[str, tuple[float, ...]], dict[str, Any]]:
    ids = sorted(FIG2_NODES)
    vectors = {k: FIG2_NODES[k]["vector"] for k in ids}
    intervals = {k: FIG2_NODES[k]["interval"] for k in ids}
    return ids, vectors, intervals


def ug_demo() -> dict[str, Any]:
    ids, vectors, intervals = _fig2_bundle()
    graph = build_ug(
        ids,
        vectors,
        intervals,
        ef_spatial=len(ids),
        ef_attribute=32,
        max_edges_if=64,
        max_edges_is=64,
        iterations=3,
    )
    bits = {
        f"{a}-{b}": (edge_mask(graph, a, b) & 1, (edge_mask(graph, a, b) >> 1) & 1)
        for a, b in EXAMPLE1_BITS
    }
    return {
        "node_count": len(ids),
        "edge_pairs": sum(len(v) for v in graph.values()) // 2,
        "example1_bits": bits,
    }


def query_demo() -> dict[str, Any]:
    ids, vectors, intervals = _fig2_bundle()
    graph = build_ug(
        ids,
        vectors,
        intervals,
        ef_spatial=len(ids),
        ef_attribute=32,
        max_edges_if=64,
        max_edges_is=64,
        iterations=3,
    )
    qvec = vectors["G"]
    qint = intervals["G"]
    from ltx_trainer.urng.intervals import Interval, QueryType

    bf_if = brute_force_query(vectors, intervals, qvec, qint, QueryType.IFANN, k=3)
    ug_if = interval_aware_beam_search(
        graph,
        vectors,
        intervals,
        qvec,
        qint,
        QueryType.IFANN,
        k=3,
        ef_search=32,
    )
    bf_is = brute_force_query(vectors, intervals, qvec, qint, QueryType.ISANN, k=3)
    ug_is = interval_aware_beam_search(
        graph,
        vectors,
        intervals,
        qvec,
        qint,
        QueryType.ISANN,
        k=3,
        ef_search=32,
    )
    rs_int = Interval(50, 50)
    bf_rs = brute_force_query(vectors, intervals, qvec, rs_int, QueryType.RSANN, k=3)
    ug_rs = interval_aware_beam_search(
        graph,
        vectors,
        intervals,
        qvec,
        rs_int,
        QueryType.RSANN,
        k=3,
        ef_search=32,
    )
    arr = build_entry_arrays(intervals)
    entry_if = get_entry_node(arr, qint, QueryType.IFANN)
    entry_is = get_entry_node(arr, qint, QueryType.ISANN)
    return {
        "query_interval": [qint.left, qint.right],
        "entry_if": entry_if,
        "entry_is": entry_is,
        "bf_if_top3": bf_if,
        "ug_if_top3": ug_if,
        "ug_matches_bf_if": ug_if == bf_if,
        "bf_is_top3": bf_is,
        "ug_is_top3": ug_is,
        "ug_matches_bf_is": ug_is == bf_is,
        "rsann_interval": [rs_int.left, rs_int.right],
        "bf_rs_top3": bf_rs,
        "ug_rs_top3": ug_rs,
        "ug_matches_bf_rs": ug_rs == bf_rs,
    }


def framework_card(cfg: UrngConfig | None = None) -> dict[str, Any]:
    cfg = cfg or UrngConfig()
    return {
        "paper": benchmarks_bundle()["paper"],
        "method": {
            "model": "URNG unified interval-aware RNG with IF/IS semantic bitmasks",
            "index": "UG — candidate gen + unified prune + iterative repair",
            "queries": ["IFANN", "ISANN", "RFANN", "RSANN"],
            "properties": ["monotonic searchability", "structural heredity"],
            "query_algo": ["GetEntryNode (Alg. 5)", "ContextAwareSearch (Alg. 4)"],
        },
        "config": cfg.__dict__,
        "integration": integration_bundle(),
    }


def evaluation_demo(*, seed: int = 0) -> dict[str, Any]:
    _ = seed
    return {
        "fig2": fig2_dataset(),
        "ug": ug_demo(),
        "query": query_demo(),
        "ref_datasets": PAPER_ANCHORS["datasets"],
        "query_types": PAPER_ANCHORS["query_types"],
    }


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    demo = evaluation_demo(seed=seed)
    ex_bits = demo["ug"]["example1_bits"]
    expected = {f"{a}-{b}": v for (a, b), v in EXAMPLE1_BITS.items()}
    bits_ok = all(ex_bits.get(k) == v for k, v in expected.items())
    return {
        "package": "urng",
        "paper": "urng",
        "arxiv": "2606.11789",
        "ref_datasets": demo["ref_datasets"],
        "query_types": demo["query_types"],
        "example1_bits_match": bits_ok,
        "ug_matches_bruteforce_if": demo["query"]["ug_matches_bf_if"],
        "ug_matches_bruteforce_is": demo["query"]["ug_matches_bf_is"],
        "ug_matches_bruteforce_rs": demo["query"]["ug_matches_bf_rs"],
        "entry_nodes_found": demo["query"]["entry_if"] is not None
        and demo["query"]["entry_is"] is not None,
        "gopex_stub_count": len(integration_bundle()["gopex_stubs"]),
    }
