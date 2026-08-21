"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cs_nmg.config import CsNmgConfig
from ltx_trainer.cs_nmg.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table2_main_results,
    table3_filter_ablations,
)


def evaluation_smoke(cfg: CsNmgConfig | None = None) -> dict[str, Any]:
    c = cfg or CsNmgConfig()
    demo = evaluation_demo(seed=0, cfg=c)
    best = next(r for r in table2_main_results() if r["method"] == "WCE + CL (tri-level)")
    ce = next(r for r in table2_main_results() if r["method"] == "CE")
    assert best["cmn_wer"] < ce["cmn_wer"]
    assert best["vie_pier"] < ce["vie_pier"]
    assert demo["losses"]["anchor_beats_negatives"]
    assert demo["near_miss"]["tri_level_kept"] >= 1

    tri = next(r for r in table3_filter_ablations() if r["group"] == "Ac.+Ph.+Text")
    assert tri["cmn_wer"] == c.cmn_wer

    b = benchmarks_bundle()
    assert len(b["table2_main_results"]) == 6

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "cmn_wer": c.cmn_wer,
        "vie_pier": c.vie_pier,
        "wer_gain_cmn_pct": c.cmn_wer_ce - c.cmn_wer,
    }
