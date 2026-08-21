"""CubePart framework card and evaluation demos."""
from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.cubepart.benchmarks import benchmarks_bundle
from ltx_trainer.cubepart.config import CubePartConfig
from ltx_trainer.cubepart.core import part_schema_meshes

def framework_card(cfg: CubePartConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CubePartConfig()
    return {
        "name": cfg.name,
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "website": cfg.website,
        "ltx_hook": "Semantic part schemas for character/object LTX control",
    }

def knowledge_card(cfg: CubePartConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CubePartConfig()
    return {"framework": framework_card(cfg), "benchmarks": benchmarks_bundle()}

def evaluation_demo(cfg: CubePartConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CubePartConfig()
    rng = np.random.default_rng(7)
    parts = part_schema_meshes(["body", "wheel_fl", "wheel_fr"], seed=3)
    return {"n_parts": len(parts), "paper_tables": benchmarks_bundle()}

def evaluation_smoke(cfg: CubePartConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CubePartConfig()
    ev = evaluation_demo(cfg)
    return {
        "package": "ltx_trainer.cubepart",
        "status": "smoke_ok",
        "paper": cfg.paper_arxiv,
        "ok": True,
        "keys": list(ev.keys())[:6],
    }
