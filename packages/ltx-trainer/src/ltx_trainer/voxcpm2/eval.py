"""End-to-end VoxCPM2 pipeline demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.voxcpm2.config import Voxcpm2Config
from ltx_trainer.voxcpm2.pipeline import evaluation_demo, headline_results, table4_inference_recipes


def pipeline_demo(*, seed: int = 0, cfg: Voxcpm2Config | None = None) -> dict[str, Any]:
    c = cfg or Voxcpm2Config()
    demo = evaluation_demo(seed=seed, cfg=c)
    best = max(table4_inference_recipes(), key=lambda r: r["en_sim"])
    return {
        "demo": demo,
        "best_recipe": best["recipe"],
        "best_en_sim": best["en_sim"],
        "beats_voxcpm1_zh_sim": c.seed_zh_sim > 77.0,
        "headline": headline_results(c),
    }


def eval_smoke() -> dict[str, Any]:
    out = pipeline_demo(seed=42)
    assert out["best_recipe"] == "Reference + Continuation"
    assert out["best_en_sim"] == 79.5
    return {"status": "ok", "internal_30lang_avg_wer": out["headline"]["internal_30lang_avg_wer"]}
