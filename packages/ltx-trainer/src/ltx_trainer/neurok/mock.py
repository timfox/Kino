"""NEUROK evaluation smoke for paper-stub validation."""

from __future__ import annotations

from typing import Any

from ltx_trainer.smoke_util import load_sibling


def evaluation_smoke(cfg=None) -> dict[str, Any]:
    config_mod = load_sibling(__file__, "config")
    pipeline_mod = load_sibling(__file__, "pipeline")
    c = cfg or config_mod.NeurokConfig()
    demo = pipeline_mod.evaluation_demo(c)
    return {
        "paper": c.paper_arxiv,
        "vae_total": float(demo["vae_losses"]["total"]),
        "ik_chamfer_l1_smoke": float(demo["ik_smoke"]["chamfer_l1"]),
        "trajectory_len": int(demo["trajectory_len"]),
        "paper_ik_chamfer_l1": float(demo["paper_ik_chamfer_l1"]),
        "user_alignment_pct": float(demo["user_study"]["alignment_pct"]),
    }
