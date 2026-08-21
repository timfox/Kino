"""SCOPE framework card and evaluation demos."""
from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.scope.benchmarks import benchmarks_bundle
from ltx_trainer.scope.config import SCOPEConfig
from ltx_trainer.scope.core import scope_action_response

def framework_card(cfg: SCOPEConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SCOPEConfig()
    return {
        "name": cfg.name,
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "website": cfg.website,
        "ltx_hook": "CrossFPS 10-DoF telemetry for interactive LTX game footage",
    }

def knowledge_card(cfg: SCOPEConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SCOPEConfig()
    return {"framework": framework_card(cfg), "benchmarks": benchmarks_bundle()}

def evaluation_demo(cfg: SCOPEConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SCOPEConfig()
    rng = np.random.default_rng(7)
    resp = scope_action_response(
        rng.standard_normal((16, 8)),
        in_scope_mask=np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
    )
    return {"in_scope_energy": resp["in_scope"], "paper_tables": benchmarks_bundle()}

def evaluation_smoke(cfg: SCOPEConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SCOPEConfig()
    ev = evaluation_demo(cfg)
    return {
        "package": "ltx_trainer.scope",
        "status": "smoke_ok",
        "paper": cfg.paper_arxiv,
        "ok": True,
        "keys": list(ev.keys())[:6],
    }
