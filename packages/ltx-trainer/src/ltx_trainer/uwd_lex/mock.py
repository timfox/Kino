"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.uwd_lex.config import UwdLexConfig
from ltx_trainer.uwd_lex.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    fig6_synthetic_scores,
)


def evaluation_smoke(cfg: UwdLexConfig | None = None) -> dict[str, Any]:
    c = cfg or UwdLexConfig()
    demo = evaluation_demo(seed=0, cfg=c)

    pure_nes = next(
        r for r in fig6_synthetic_scores() if r["lexicon"] == "large-pure" and r["metric"] == "NES"
    )
    impure_nes = next(
        r for r in fig6_synthetic_scores() if r["lexicon"] == "large-impure" and r["metric"] == "NES"
    )
    assert pure_nes["score_pct"] == c.synth_large_pure_nes_pct
    assert impure_nes["score_pct"] == c.synth_large_impure_nes_pct
    assert pure_nes["score_pct"] > impure_nes["score_pct"]

    pure_wnes = next(
        r for r in fig6_synthetic_scores() if r["lexicon"] == "large-pure" and r["metric"] == "WNES"
    )
    impure_wnes = next(
        r for r in fig6_synthetic_scores() if r["lexicon"] == "large-impure" and r["metric"] == "WNES"
    )
    assert impure_wnes["score_pct"] > pure_wnes["score_pct"]

    assert demo["f1_wnes_balances_extremes"]
    assert demo["beats_ned_bitrate_tradeoff"]
    assert len(benchmarks_bundle()["fig4_real_world"]) == 3

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "best_system": c.best_system,
        "best_k": c.best_k,
        "synth_nes_gap_pct": c.synth_large_pure_nes_pct - c.synth_large_impure_nes_pct,
        "synth_wnes_reversal": c.synth_large_impure_wnes_pct > c.synth_large_pure_wnes_pct,
    }
