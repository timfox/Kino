"""PhyWorld flow-matching + DPO toy smoke (arXiv:2605.19242)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.smoke_util import load_sibling


def evaluation_smoke(cfg=None) -> dict[str, Any]:
    config_mod = load_sibling(__file__, "config")
    pipeline_mod = load_sibling(__file__, "pipeline")
    c = cfg or config_mod.PhyWorldConfig()
    demo = pipeline_mod.pipeline_demo(c)
    return {
        "paper": c.paper_arxiv,
        "flow_matching_loss": round(float(demo["flow_matching_loss"]), 4),
        "dpo_loss": round(float(demo["dpo_loss"]), 4),
        "preference_margin_ok": bool(demo["preference_margin_ok"]),
        "v2v_channel_count": int(demo["v2v_channel_count"]),
        "overall_score_toy": round(float(demo["overall_score_toy"]), 2),
    }
