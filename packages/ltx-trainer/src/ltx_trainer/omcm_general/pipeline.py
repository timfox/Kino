"""Pipeline for OMCM general arrivals."""

from __future__ import annotations

from typing import Any

from ltx_trainer.omcm_general.config import OmcmGeneralConfig
from ltx_trainer.omcm_general.matching import (
    competitive_ratio_bound,
    simulate_general_arrivals,
    toy_arrival_stream,
)


def framework_card(cfg: OmcmGeneralConfig | None = None) -> dict[str, Any]:
    cfg = cfg or OmcmGeneralConfig()
    return {
        "name": "OMCM-General",
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "model": "non-bipartite perfect matching on [0,1] line, general arrivals",
        "results": {
            "unknown_iid": cfg.unknown_iid_cr,
            "random_order": cfg.random_order_cr,
        },
    }


def table_main_results() -> list[dict[str, str]]:
    return [
        {"setting": "Non-bipartite, unknown i.i.d.", "cr": "O(log² n)", "algorithm": "primal-dual + dual fitting"},
        {"setting": "Non-bipartite, random order", "cr": "unbounded", "algorithm": "lower bound (separation)"},
        {"setting": "Classic one-sided (baseline)", "cr": "O(log n)", "algorithm": "prior work"},
    ]


def benchmarks_bundle() -> dict[str, Any]:
    return {"main_results": table_main_results()}


def evaluation_demo(*, seed: int = 0, n: int = 20) -> dict[str, Any]:
    stream = toy_arrival_stream(seed, n=n)
    sim = simulate_general_arrivals(stream)
    return {
        "paper": OmcmGeneralConfig().paper_arxiv,
        "simulation": sim,
        "cr_bound_unknown_iid": competitive_ratio_bound(n, model="unknown_iid"),
        "cr_random_order": competitive_ratio_bound(n, model="random_order"),
    }
