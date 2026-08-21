"""RGB–NIR IR evaluation smoke for paper-stub validation."""

from __future__ import annotations

from typing import Any

from ltx_trainer.smoke_util import load_sibling


def evaluation_smoke(cfg=None) -> dict[str, Any]:
    config_mod = load_sibling(__file__, "config")
    pipeline_mod = load_sibling(__file__, "pipeline")
    c = cfg or config_mod.RgbNirIrConfig()
    demo = pipeline_mod.evaluation_demo(c)
    return {
        "paper": c.paper_arxiv,
        "stage2_basis_mse": float(demo["stage2_basis"]["final_mse"]),
        "paper_albedo_psnr": float(demo["paper_albedo_psnr"]),
        "psnr_gain_vs_wildlight": float(demo["psnr_gain_vs_wildlight"]),
    }
