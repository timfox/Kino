"""PhysX-Omni framework card and evaluation demos."""
from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.physx_omni.benchmarks import benchmarks_bundle
from ltx_trainer.physx_omni.config import PhysXOmniConfig
from ltx_trainer.physx_omni.core import physx_bench_scores

def framework_card(cfg: PhysXOmniConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PhysXOmniConfig()
    return {
        "name": cfg.name,
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "website": cfg.website,
        "ltx_hook": "PhysXVerse assets for physics-steering and world-model eval",
    }

def knowledge_card(cfg: PhysXOmniConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PhysXOmniConfig()
    return {"framework": framework_card(cfg), "benchmarks": benchmarks_bundle()}

def evaluation_demo(cfg: PhysXOmniConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PhysXOmniConfig()
    rng = np.random.default_rng(7)
    scores = physx_bench_scores(
        geometry=0.9,
        material=0.85,
        kinematics=0.88,
    )
    return {"bench": scores, "paper_tables": benchmarks_bundle()}

def evaluation_smoke(cfg: PhysXOmniConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PhysXOmniConfig()
    ev = evaluation_demo(cfg)
    return {
        "package": "ltx_trainer.physx_omni",
        "status": "smoke_ok",
        "paper": cfg.paper_arxiv,
        "ok": True,
        "keys": list(ev.keys())[:6],
    }
