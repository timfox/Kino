"""SEGA framework card and evaluation demos."""
from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.sega.benchmarks import benchmarks_bundle
from ltx_trainer.sega.config import SEGAConfig
from ltx_trainer.sega.core import sega_rope_scales

def framework_card(cfg: SEGAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SEGAConfig()
    return {
        "name": cfg.name,
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "website": cfg.website,
        "ltx_hook": "Training-free HQ 1088p+ LTX inference without retrain",
    }

def knowledge_card(cfg: SEGAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SEGAConfig()
    return {"framework": framework_card(cfg), "benchmarks": benchmarks_bundle()}

def evaluation_demo(cfg: SEGAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SEGAConfig()
    rng = np.random.default_rng(7)
    scales = sega_rope_scales(rng.standard_normal((64, 64)), base_scale=1.0)
    return {"scale_std": float(np.std(scales)), "paper_tables": benchmarks_bundle()}

def evaluation_smoke(cfg: SEGAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SEGAConfig()
    ev = evaluation_demo(cfg)
    return {
        "package": "ltx_trainer.sega",
        "status": "smoke_ok",
        "paper": cfg.paper_arxiv,
        "ok": True,
        "keys": list(ev.keys())[:6],
    }
