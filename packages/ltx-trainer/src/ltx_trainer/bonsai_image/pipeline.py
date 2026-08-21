"""Bonsai Image 4B framework card and evaluation demos."""
from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.bonsai_image.benchmarks import benchmarks_bundle
from ltx_trainer.bonsai_image.config import BonsaiImageConfig
from ltx_trainer.bonsai_image.core import bonsai_weight_stats

def framework_card(cfg: BonsaiImageConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BonsaiImageConfig()
    return {
        "name": cfg.name,
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "website": cfg.website,
        "ltx_hook": "Local still-frame reference gen alongside LTX keyframes",
    }

def knowledge_card(cfg: BonsaiImageConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BonsaiImageConfig()
    return {"framework": framework_card(cfg), "benchmarks": benchmarks_bundle()}

def evaluation_demo(cfg: BonsaiImageConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BonsaiImageConfig()
    rng = np.random.default_rng(7)
    stats = bonsai_weight_stats(bits=1.125)
    return {"variant": "1-bit", "transformer_gb": stats["gb"], "paper_tables": benchmarks_bundle()}

def evaluation_smoke(cfg: BonsaiImageConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BonsaiImageConfig()
    ev = evaluation_demo(cfg)
    return {
        "package": "ltx_trainer.bonsai_image",
        "status": "smoke_ok",
        "paper": cfg.paper_arxiv,
        "ok": True,
        "keys": list(ev.keys())[:6],
    }
