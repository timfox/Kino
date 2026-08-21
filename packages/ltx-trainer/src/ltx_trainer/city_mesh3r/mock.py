"""City-Mesh3R evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.smoke_util import load_sibling


def evaluation_smoke(cfg=None) -> dict[str, Any]:
    config_mod = load_sibling(__file__, "config")
    pipeline_mod = load_sibling(__file__, "pipeline")
    c = cfg or config_mod.CityMesh3RConfig()
    demo = pipeline_mod.evaluation_demo(c)
    return {
        "paper": c.paper_arxiv,
        "num_clusters": int(demo["num_clusters"]),
        "paper_f1_cuhk_lower": float(demo["paper_f1_cuhk_lower"]),
        "paper_sfm_hours": float(demo["paper_sfm_hours"]),
        "mesh_total_loss": float(demo["mesh_losses"]["total"]),
    }
