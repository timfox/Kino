"""DEMON evaluation smoke (arXiv:2605.28657)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.demon.config import DemonConfig
from ltx_trainer.demon.pipeline import pipeline_demo
from ltx_trainer.demon.ring_buffer import ablation_completion_rates


def evaluation_smoke(cfg: DemonConfig | None = None) -> dict[str, Any]:
    c = cfg or DemonConfig()
    demo = pipeline_demo(c, seed=42)
    ab = ablation_completion_rates(c)
    return {
        "paper": c.paper_arxiv,
        "gens_per_sec_depth8": c.gens_per_sec_depth8,
        "per_slot_beats_reset": ab["per_slot_sweep"] > ab["global_reset_sweep"],
        "vae_speedup_3s": demo["vae"]["speedup_3s"],
        "n_per_frame_curves": demo["sde"]["n_curves"],
        "heterogeneous_ring": demo["ring"]["heterogeneous_timesteps"],
    }
