"""BES evaluation demos and smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.bes.benchmarks import benchmarks_bundle
from ltx_trainer.bes.config import BESConfig
from ltx_trainer.bes.core import run_toy_search
from ltx_trainer.bes.experiments import experiments_bundle
from ltx_trainer.bes.theory import theory_card


def evaluation_demo(cfg: BESConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BESConfig()
    toy = run_toy_search(seed=7)
    return {
        "paper": cfg.paper_arxiv,
        "toy_search": toy["search"],
        "operators_sample": {
            k: list(v) if isinstance(v, tuple) else v
            for k, v in toy["operators"].items()
            if k != "path_a"
        },
        "theory": theory_card(),
        "paper_tables": benchmarks_bundle(),
        "experiments": experiments_bundle(),
    }


def evaluation_smoke(cfg: BESConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BESConfig()
    ev = evaluation_demo(cfg)
    ok = ev["toy_search"]["best_score"] > 0.5
    return {
        "package": "ltx_trainer.bes",
        "status": "smoke_ok" if ok else "smoke_weak",
        "paper": cfg.paper_arxiv,
        "ok": ok,
        "best_score": ev["toy_search"]["best_score"],
        "keys": list(ev.keys()),
    }
