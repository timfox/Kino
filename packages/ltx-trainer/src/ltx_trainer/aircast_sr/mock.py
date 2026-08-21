"""Runnable evaluation smoke for AirCast-SR (arXiv:2605.26130)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.aircast_sr.config import AirCastSRConfig
from ltx_trainer.aircast_sr.pipeline import benchmarks_bundle, evaluation_demo, framework_card, headline_results


def evaluation_smoke() -> dict[str, Any]:
    cfg = AirCastSRConfig()
    demo = evaluation_demo(seed=0)
    card = framework_card(cfg)
    bench = benchmarks_bundle()
    head = headline_results()
    return {
        "paper": cfg.paper_arxiv,
        "framework": card["name"],
        "train_mse": demo["train_mse"],
        "t2m_r": demo["t2m_skill"]["r"],
        "t2m_bias": demo["t2m_skill"]["bias"],
        "psd_bins": demo["psd_bins"],
        "patch_merge_ok": demo["merge_shape"][-2:] == [16, 16],
        "lcm_steps": demo["lcm_steps"],
        "table_count": len(bench),
        "winter_t2m_r_6h": head["winter_t2m_r_6h"],
        "india_t2m_r_48h": head["india_t2m_r_48h"],
        "in_channels": cfg.denoiser_in_channels,
        "target_variables": cfg.n_target_variables,
    }
