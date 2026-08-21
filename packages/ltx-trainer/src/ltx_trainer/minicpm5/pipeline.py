"""MiniCPM5-1B framework card and evaluation demos."""
from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.minicpm5.benchmarks import benchmarks_bundle
from ltx_trainer.minicpm5.config import MiniCPM5Config

def framework_card(cfg: MiniCPM5Config | None = None) -> dict[str, Any]:
    cfg = cfg or MiniCPM5Config()
    return {
        "name": cfg.name,
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "website": cfg.website,
        "ltx_hook": "Lightweight caption/VLM sidecar when Gemma4 is busy on GPU 1",
    }

def knowledge_card(cfg: MiniCPM5Config | None = None) -> dict[str, Any]:
    cfg = cfg or MiniCPM5Config()
    return {"framework": framework_card(cfg), "benchmarks": benchmarks_bundle()}

def evaluation_demo(cfg: MiniCPM5Config | None = None) -> dict[str, Any]:
    cfg = cfg or MiniCPM5Config()
    rng = np.random.default_rng(7)
    return {"hub": "openbmb/MiniCPM5-1B", "role": "edge_caption_vlm", "paper_tables": benchmarks_bundle()}

def evaluation_smoke(cfg: MiniCPM5Config | None = None) -> dict[str, Any]:
    cfg = cfg or MiniCPM5Config()
    ev = evaluation_demo(cfg)
    return {
        "package": "ltx_trainer.minicpm5",
        "status": "smoke_ok",
        "paper": cfg.paper_arxiv,
        "ok": True,
        "keys": list(ev.keys())[:6],
    }
