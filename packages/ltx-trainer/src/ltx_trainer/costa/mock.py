"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.costa.config import CostaConfig
from ltx_trainer.costa.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table3_tta,
    table4_comparisons,
)


def evaluation_smoke(cfg: CostaConfig | None = None) -> dict[str, Any]:
    c = cfg or CostaConfig()
    demo = evaluation_demo(seed=0, cfg=c)

    costa = next(r for r in table4_comparisons() if r["method"] == "CoSTA (Ours)")
    assert costa["accuracy_pct"] == c.best_accuracy_pct
    assert round(c.best_accuracy_pct - c.baseline_accuracy_pct, 2) == round(
        c.gain_over_baseline_pct, 2
    )

    tta_best = max(table3_tta(), key=lambda r: r["with_tta_pct"])
    assert tta_best["with_tta_pct"] == c.tta_w2v960_large_lv_pct

    assert demo["beats_baseline"]
    assert demo["optimal_aug_factor"] == c.optimal_aug_factor
    assert len(benchmarks_bundle()["table1_tts_objective"]) == 4

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "baseline_accuracy_pct": c.baseline_accuracy_pct,
        "best_accuracy_pct": c.best_accuracy_pct,
        "gain_over_baseline_pct": c.gain_over_baseline_pct,
        "cs_cosy_beat_baseline_ratio": c.cs_cosy_beat_baseline_ratio,
    }
