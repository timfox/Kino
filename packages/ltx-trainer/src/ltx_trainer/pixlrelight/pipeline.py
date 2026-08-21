"""PIXLRelight framework card and evaluation demos."""
from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.pixlrelight.benchmarks import benchmarks_bundle
from ltx_trainer.pixlrelight.config import PIXLRelightConfig
from ltx_trainer.pixlrelight.core import intrinsic_conditioning_pack

def framework_card(cfg: PIXLRelightConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PIXLRelightConfig()
    return {
        "name": cfg.name,
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "website": cfg.website,
        "ltx_hook": "Affine-modulated transformer relight for LTX HDR bracket plans",
    }

def knowledge_card(cfg: PIXLRelightConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PIXLRelightConfig()
    return {"framework": framework_card(cfg), "benchmarks": benchmarks_bundle()}

def evaluation_demo(cfg: PIXLRelightConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PIXLRelightConfig()
    rng = np.random.default_rng(7)
    intr = intrinsic_conditioning_pack(rng.random((32, 32, 3)))
    return {"albedo_mean": float(intr["albedo"].mean()), "paper_tables": benchmarks_bundle()}

def evaluation_smoke(cfg: PIXLRelightConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PIXLRelightConfig()
    ev = evaluation_demo(cfg)
    return {
        "package": "ltx_trainer.pixlrelight",
        "status": "smoke_ok",
        "paper": cfg.paper_arxiv,
        "ok": True,
        "keys": list(ev.keys())[:6],
    }
