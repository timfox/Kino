"""CAFNet evaluation smoke (arXiv:2605.29531)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cafnet.config import CafNetConfig
from ltx_trainer.cafnet.pipeline import pipeline_demo


def evaluation_smoke(cfg: CafNetConfig | None = None) -> dict[str, Any]:
    c = cfg or CafNetConfig()
    demo = pipeline_demo(c, seed=c.random_seed)
    return {
        "paper": c.paper_arxiv,
        "params": c.cafnet_params,
        "feature_shapes": demo["feature_shapes"],
        "loss_total": demo["loss_half_truth"]["total"],
        "boundary_mae_s": demo["boundary_eval"]["boundary_mae_s"],
    }
