"""End-to-end Eisbach pipeline demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.eisbach.config import EisbachConfig
from ltx_trainer.eisbach.pipeline import evaluation_demo, table2_structural_dimensions


def pipeline_demo(*, seed: int = 0, cfg: EisbachConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EisbachConfig()
    demo = evaluation_demo(seed=seed, cfg=cfg)
    dyn = next(r for r in table2_structural_dimensions() if r["dimension"] == "Dynamic range")
    return {
        "demo": demo,
        "barrier_lambda": cfg.barrier_lambda,
        "barrier_dynamic_range": dyn["barrier"],
        "baseline_dynamic_range": dyn["baseline"],
        "develops_not_repeats": demo["analysis"]["barrier_more_development"],
    }


def eval_smoke() -> dict[str, Any]:
    out = pipeline_demo(seed=42)
    assert out["barrier_lambda"] == 0.5
    assert out["develops_not_repeats"]
    assert ">40" in out["barrier_dynamic_range"]
    return {"status": "ok", "barrier_lambda": out["barrier_lambda"]}
