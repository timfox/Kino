"""Claude Opus 4.8 framework card and evaluation demos."""
from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.opus48.benchmarks import benchmarks_bundle
from ltx_trainer.opus48.config import Opus48Config

def framework_card(cfg: Opus48Config | None = None) -> dict[str, Any]:
    cfg = cfg or Opus48Config()
    return {
        "name": cfg.name,
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "website": cfg.website,
        "ltx_hook": "DeepSWE / agent harness reference checkpoint",
    }

def knowledge_card(cfg: Opus48Config | None = None) -> dict[str, Any]:
    cfg = cfg or Opus48Config()
    return {"framework": framework_card(cfg), "benchmarks": benchmarks_bundle()}

def evaluation_demo(cfg: Opus48Config | None = None) -> dict[str, Any]:
    cfg = cfg or Opus48Config()
    rng = np.random.default_rng(7)
    return {"ok": True, "paper_tables": benchmarks_bundle()}

def evaluation_smoke(cfg: Opus48Config | None = None) -> dict[str, Any]:
    cfg = cfg or Opus48Config()
    ev = evaluation_demo(cfg)
    return {
        "package": "ltx_trainer.opus48",
        "status": "smoke_ok",
        "paper": cfg.paper_arxiv,
        "ok": True,
        "keys": list(ev.keys())[:6],
    }
