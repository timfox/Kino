"""Evaluation smoke for resfree_geo paper stub."""

from __future__ import annotations

from typing import Any

from ltx_trainer.smoke_util import load_sibling


def evaluation_smoke(cfg=None) -> dict[str, Any]:
    config_mod = load_sibling(__file__, "config")
    pipeline_mod = load_sibling(__file__, "pipeline")
    c = cfg or config_mod.ResfreeGeoConfig()
    demo = pipeline_mod.evaluation_demo(c)
    return {
        "paper": c.paper_arxiv,
        "beltrami_energy": float(demo["beltrami_energy"]),
        "table2_case_a_delta_mu": float(demo["table2_case_a_delta_mu"]),
        "table4_ring_std_map": float(demo["table4_ring_std_map"]),
    }
