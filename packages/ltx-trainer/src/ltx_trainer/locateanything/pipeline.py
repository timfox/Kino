"""LocateAnything framework card and evaluation demos."""
from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.locateanything.benchmarks import benchmarks_bundle
from ltx_trainer.locateanything.config import LocateAnythingConfig
from ltx_trainer.locateanything.core import parallel_box_decode

def framework_card(cfg: LocateAnythingConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LocateAnythingConfig()
    return {
        "name": cfg.name,
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "website": cfg.website,
        "ltx_hook": "Shot-level bbox metadata for LTX caption QA and UI/document clips",
    }

def knowledge_card(cfg: LocateAnythingConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LocateAnythingConfig()
    return {"framework": framework_card(cfg), "benchmarks": benchmarks_bundle()}

def evaluation_demo(cfg: LocateAnythingConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LocateAnythingConfig()
    rng = np.random.default_rng(7)
    boxes = parallel_box_decode(
        rng.standard_normal((4, 4)),
        threshold=0.5,
    )
    return {
        "n_boxes": len(boxes),
        "bps_stub": 12.7,
        "mean_score": float(np.mean([b["score"] for b in boxes])),
        "paper_tables": benchmarks_bundle(),
    }

def evaluation_smoke(cfg: LocateAnythingConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LocateAnythingConfig()
    ev = evaluation_demo(cfg)
    return {
        "package": "ltx_trainer.locateanything",
        "status": "smoke_ok",
        "paper": cfg.paper_arxiv,
        "ok": True,
        "keys": list(ev.keys())[:6],
    }
