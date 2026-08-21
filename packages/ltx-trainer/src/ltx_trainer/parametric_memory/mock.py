"""Evaluation smoke for parametric_memory stub."""

from __future__ import annotations

from typing import Any

from ltx_trainer.smoke_util import load_sibling


def evaluation_smoke(cfg=None) -> dict[str, Any]:
    config_mod = load_sibling(__file__, "config")
    pipeline_mod = load_sibling(__file__, "pipeline")
    c = cfg or config_mod.ParametricMemoryConfig()
    demo = pipeline_mod.evaluation_demo(c)
    return {
        "paper": c.paper_arxiv,
        "l_crit": float(demo["l_crit"]),
        "loss_memft_ot": float(demo["loss_memft_ot"]),
        "qwen_memft_ot_r9_acctok": float(demo["qwen_memft_ot_r9_acctok"]),
        "table1_qwen_comb_r2": float(demo["table1_qwen_comb_r2"]),
    }
