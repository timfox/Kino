"""Pipeline and benchmarks for dynamic quasi-cliques."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dyn_quasi_clique.config import DynQuasiCliqueConfig
from ltx_trainer.dyn_quasi_clique.graph import (
    dynamic_update_smoke,
    edge_density,
    fast_nbsim_seed,
    toy_dynamic_graph,
)


def framework_card(cfg: DynQuasiCliqueConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DynQuasiCliqueConfig()
    return {
        "name": "DynQuasiClique",
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "static_baseline": "FastNBSim (Pang et al., WWW 2024)",
        "variants": ["incremental credit-based", "fully dynamic"],
        "metrics": ["quasi-clique size", "update time vs rerun NBSim"],
        "alpha": cfg.alpha_default,
        "gamma": cfg.gamma_default,
    }


def table_speedups(cfg: DynQuasiCliqueConfig | None = None) -> dict[str, float]:
    cfg = cfg or DynQuasiCliqueConfig()
    return {
        "incremental_vs_rerun": cfg.incremental_speedup_anchor,
        "fully_dynamic_vs_rerun": cfg.fully_dynamic_speedup_anchor,
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "speedup_anchors": table_speedups(),
        "datasets": ["DBLP co-authorship snapshots", "social network traces"],
    }


def evaluation_demo(*, seed: int = 0) -> dict[str, Any]:
    adj = toy_dynamic_graph(seed)
    clique = fast_nbsim_seed(adj, alpha=0.85, gamma=0.5)
    updates = [(0, 9, True), (1, 10, True), (2, 11, False)]
    dyn = dynamic_update_smoke(dict(adj), updates, alpha=0.85)
    return {
        "paper": DynQuasiCliqueConfig().paper_arxiv,
        "static_clique_size": len(clique),
        "static_density": round(edge_density(clique, adj), 4),
        "dynamic_trace": dyn["size_trace"],
    }
