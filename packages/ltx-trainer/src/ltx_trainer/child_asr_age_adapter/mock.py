"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.child_asr_age_adapter.config import ChildAsrAgeAdapterConfig
from ltx_trainer.child_asr_age_adapter.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table2_wer_results,
)


def evaluation_smoke(cfg: ChildAsrAgeAdapterConfig | None = None) -> dict[str, Any]:
    c = cfg or ChildAsrAgeAdapterConfig()
    demo = evaluation_demo(seed=0, cfg=c)

    best = next(r for r in table2_wer_results(c) if r.get("best"))
    assert best["wer"] == c.best_wer
    assert best["macro_wer"] == c.best_macro_wer

    shared = next(r for r in table2_wer_results(c) if r["method"] == "Child shared (db=128)")
    assert shared["wer"] == c.shared_wer
    assert best["wer"] < shared["wer"]

    film = next(r for r in table2_wer_results(c) if "FiLM (hom., GT)" in r["method"])
    assert best["wer"] < film["wer"]

    assert demo["age_specialized_beats_shared"]
    assert demo["age_specialized_beats_film"]
    assert demo["pt_top2_matches_gt_wer"]
    assert c.router_accuracy > 0.74
    assert len(benchmarks_bundle()["table1_dataset_stats"]) == 3

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "shared_wer": c.shared_wer,
        "best_wer": c.best_wer,
        "best_macro_wer": c.best_macro_wer,
        "router_accuracy": c.router_accuracy,
    }
