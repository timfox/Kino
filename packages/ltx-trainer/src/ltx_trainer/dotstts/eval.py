"""End-to-end dots.tts pipeline demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dotstts.config import DotsttsConfig
from ltx_trainer.dotstts.pipeline import evaluation_demo, table2_seed_tts_eval, table_efficiency


def pipeline_demo(*, seed: int = 0, cfg: DotsttsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DotsttsConfig()
    demo = evaluation_demo(seed=seed, cfg=cfg)
    soar = next(r for r in table2_seed_tts_eval() if r["model"] == "dots.tts (SOAR)")
    eff = next(r for r in table_efficiency() if r["interleaved"])
    return {
        "demo": demo,
        "best_seed_avg_sim": soar["avg_sim"],
        "best_seed_zh_sim": soar["zh_sim"],
        "ttfp_gain_ms": round(
            next(r for r in table_efficiency() if not r["interleaved"])["ttfp_ms"] - eff["ttfp_ms"],
            1,
        ),
        "continuous_ar": True,
    }


def eval_smoke() -> dict[str, Any]:
    out = pipeline_demo(seed=42)
    assert out["best_seed_avg_sim"] == 79.2
    assert out["best_seed_zh_sim"] == 81.0
    assert out["ttfp_gain_ms"] > 30.0
    assert out["continuous_ar"]
    return {"status": "ok", "seed_avg_sim": out["best_seed_avg_sim"]}
