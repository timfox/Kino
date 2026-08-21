"""Gamma-World framework card and evaluation demos."""
from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.gamma_world.benchmarks import benchmarks_bundle
from ltx_trainer.gamma_world.config import GammaWorldConfig
from ltx_trainer.gamma_world.core import simplex_agent_phases

def framework_card(cfg: GammaWorldConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GammaWorldConfig()
    return {
        "name": cfg.name,
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "website": cfg.website,
        "ltx_hook": "24 FPS causal student for multi-agent LTX rollouts",
    }

def knowledge_card(cfg: GammaWorldConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GammaWorldConfig()
    return {"framework": framework_card(cfg), "benchmarks": benchmarks_bundle()}

def evaluation_demo(cfg: GammaWorldConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GammaWorldConfig()
    rng = np.random.default_rng(7)
    phases = simplex_agent_phases(n_agents=4)
    return {"n_agents": 4, "phase_spread": float(np.std(phases)), "paper_tables": benchmarks_bundle()}

def evaluation_smoke(cfg: GammaWorldConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GammaWorldConfig()
    ev = evaluation_demo(cfg)
    return {
        "package": "ltx_trainer.gamma_world",
        "status": "smoke_ok",
        "paper": cfg.paper_arxiv,
        "ok": True,
        "keys": list(ev.keys())[:6],
    }
