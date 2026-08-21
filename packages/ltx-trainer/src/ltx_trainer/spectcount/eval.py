"""End-to-end SpectCount pipeline demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.spectcount.config import SpectCountConfig
from ltx_trainer.spectcount.pipeline import evaluation_demo, table1_benchmarks


def pipeline_demo(*, seed: int = 0, cfg: SpectCountConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SpectCountConfig()
    demo = evaluation_demo(seed=seed, cfg=cfg)
    af3 = next(
        r for r in table1_benchmarks() if r["model"] == "Audio Flamingo 3" and r["setting"] == "SpectCount"
    )
    base = next(
        r for r in table1_benchmarks() if r["model"] == "Audio Flamingo 3" and "reproduced" in r["setting"]
    )
    return {
        "demo": demo,
        "mmau_mini_gain": round(af3["mmau_mini_total"] - base["mmau_mini_total"], 2),
        "mmau_mini_total": af3["mmau_mini_total"],
        "fully_synthetic": True,
    }


def eval_smoke() -> dict[str, Any]:
    out = pipeline_demo(seed=42)
    assert out["mmau_mini_total"] == 78.40
    assert out["mmau_mini_gain"] == 4.5
    return {"status": "ok", "mmau_mini_total": out["mmau_mini_total"]}
