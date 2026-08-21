"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mvse_amd.config import MvseAmdConfig, compute_oracle_gap_recovery_pct
from ltx_trainer.mvse_amd.pipeline import benchmarks_bundle, evaluation_demo, table3_retrieval


def evaluation_smoke(cfg: MvseAmdConfig | None = None) -> dict[str, Any]:
    c = cfg or MvseAmdConfig()
    demo = evaluation_demo(seed=0, cfg=c)

    adaptive = next(r for r in table3_retrieval(c) if r["system"] == "Adaptive +Cross")
    assert adaptive["p1"] == c.adaptive_p1

    recovered_pct = compute_oracle_gap_recovery_pct(c.adaptive_p1, c.fixed_p1, c.oracle_p1)
    assert recovered_pct == c.oracle_gap_recovery_pct

    assert c.adaptive_p1 > c.fixed_p1
    assert c.fixed_p1 < c.face_p1
    assert c.detect_cross_acc > c.detect_base_acc
    assert demo["feature_dim"] == c.feature_dim
    assert len(benchmarks_bundle()["presence_counts"]) == 3

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "adaptive_p1": c.adaptive_p1,
        "fixed_p1": c.fixed_p1,
        "oracle_p1": c.oracle_p1,
        "detection_accuracy": c.detect_full_acc,
        "oracle_gap_recovery_pct": c.oracle_gap_recovery_pct,
    }
