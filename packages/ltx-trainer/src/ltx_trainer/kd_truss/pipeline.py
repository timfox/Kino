"""Framework card, evaluation demo, and smoke entry points."""

from __future__ import annotations

from typing import Any

from ltx_trainer.kd_truss.benchmarks import PAPER_ANCHORS, benchmarks_bundle
from ltx_trainer.kd_truss.config import KdTrussConfig
from ltx_trainer.kd_truss.construction import construction_demo
from ltx_trainer.kd_truss.dc_index import dc_index_demo
from ltx_trainer.kd_truss.integration import integration_bundle
from ltx_trainer.kd_truss.maintenance import maintenance_demo
from ltx_trainer.kd_truss.running_example import fig1_temporal_graph
from ltx_trainer.kd_truss.tc_index import tc_index_demo
from ltx_trainer.kd_truss.truss import truss_demo


def framework_card(cfg: KdTrussConfig | None = None) -> dict[str, Any]:
    cfg = cfg or KdTrussConfig()
    return {
        "paper": benchmarks_bundle()["paper"],
        "method": {
            "model": "(k, δ)-truss on span-constrained δ-triangles (mts ≤ δ)",
            "query": ["Online-Query (peeling)", "TC-Query", "DC-Query"],
            "indexes": ["TC-Index (temporal containment map)", "DC-Index (dual arborescence)"],
            "construction": ["DBA (truss decomposition)", "MBA (truss maintenance)"],
            "maintenance": "Filter-and-verification on edge/timestamp insert",
            "dual_containment": "Tk,δ ⊆ Tk-1,δ and Tk,δ ⊆ Tk,δ+1",
        },
        "config": cfg.__dict__,
        "integration": integration_bundle(),
    }


def evaluation_demo(*, seed: int = 0) -> dict[str, Any]:
    _ = seed
    return {
        "fig1": fig1_temporal_graph(),
        "truss": truss_demo(),
        "tc_index": tc_index_demo(),
        "dc_index": dc_index_demo(),
        "construction": construction_demo(),
        "maintenance": maintenance_demo(),
        "ref_datasets": PAPER_ANCHORS["datasets"],
        "ref_speedup_orders": PAPER_ANCHORS["index_speedup_orders"],
    }


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    demo = evaluation_demo(seed=seed)
    return {
        "package": "kd_truss",
        "paper": "kd_truss",
        "arxiv": "2606.11582",
        "ref_datasets": demo["ref_datasets"],
        "ref_speedup_orders": demo["ref_speedup_orders"],
        "tc_matches_online": demo["tc_index"]["tc_matches_online"],
        "dc_matches_online": demo["dc_index"]["dc_matches_online"],
        "example5_core_in_truss": demo["truss"]["example5_core_in_truss"],
        "gopex_stub_count": len(integration_bundle()["gopex_stubs"]),
    }
