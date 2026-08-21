"""Relightable Holoported Characters framework card and evaluation demos."""
from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.relightable_chars.benchmarks import benchmarks_bundle
from ltx_trainer.relightable_chars.config import RelightableCharsConfig
from ltx_trainer.relightable_chars.core import texel_gaussian_count

def framework_card(cfg: RelightableCharsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RelightableCharsConfig()
    return {
        "name": cfg.name,
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "website": cfg.website,
        "ltx_hook": "RelightNet + texel-aligned 3DGS for character LTX lighting",
    }

def knowledge_card(cfg: RelightableCharsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RelightableCharsConfig()
    return {"framework": framework_card(cfg), "benchmarks": benchmarks_bundle()}

def evaluation_demo(cfg: RelightableCharsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RelightableCharsConfig()
    rng = np.random.default_rng(7)
    splats = texel_gaussian_count(mesh_texels=128)
    return {"n_splats": splats, "paper_tables": benchmarks_bundle()}

def evaluation_smoke(cfg: RelightableCharsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RelightableCharsConfig()
    ev = evaluation_demo(cfg)
    return {
        "package": "ltx_trainer.relightable_chars",
        "status": "smoke_ok",
        "paper": cfg.paper_arxiv,
        "ok": True,
        "keys": list(ev.keys())[:6],
    }
