"""GenRecon framework card and evaluation demos."""
from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.genrecon.benchmarks import benchmarks_bundle
from ltx_trainer.genrecon.config import GenReconConfig
from ltx_trainer.genrecon.core import overlapping_scene_chunks

def framework_card(cfg: GenReconConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GenReconConfig()
    return {
        "name": cfg.name,
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "website": cfg.website,
        "ltx_hook": "Indoor scene mesh proxy for LTX camera-path planning",
    }

def knowledge_card(cfg: GenReconConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GenReconConfig()
    return {"framework": framework_card(cfg), "benchmarks": benchmarks_bundle()}

def evaluation_demo(cfg: GenReconConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GenReconConfig()
    rng = np.random.default_rng(7)
    chunks = overlapping_scene_chunks(n_chunks=3, overlap=0.25)
    return {"n_chunks": len(chunks), "overlap": 0.25, "paper_tables": benchmarks_bundle()}

def evaluation_smoke(cfg: GenReconConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GenReconConfig()
    ev = evaluation_demo(cfg)
    return {
        "package": "ltx_trainer.genrecon",
        "status": "smoke_ok",
        "paper": cfg.paper_arxiv,
        "ok": True,
        "keys": list(ev.keys())[:6],
    }
