"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.speechjbb.config import SpeechJbbConfig
from ltx_trainer.speechjbb.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table3_model_results,
)


def evaluation_smoke(cfg: SpeechJbbConfig | None = None) -> dict[str, Any]:
    c = cfg or SpeechJbbConfig()
    demo = evaluation_demo(seed=0, cfg=c)

    mean_row = next(r for r in table3_model_results(c) if r.get("average"))
    assert mean_row["mono_jsr"] == c.mono_jsr
    assert mean_row["xy_jsr"] == c.xy_jsr

    voxtral = next(r for r in table3_model_results(c) if r["model"] == "Voxtral")
    gemini = next(r for r in table3_model_results(c) if r["model"] == "Gemini")
    assert voxtral["avg_jsr"] == c.voxtral_mean_jsr
    assert gemini["avg_jsr"] == c.gemini_mean_jsr

    assert c.xy_jsr > c.mono_jsr
    assert c.pseudo_50_jsr > c.mean_jsr
    assert demo["xy_jsr_exceeds_mono_anchor"]
    assert len(benchmarks_bundle()["code_switch_pairs"]) == 10

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "mono_jsr": c.mono_jsr,
        "xy_jsr": c.xy_jsr,
        "voxtral_mean_jsr": c.voxtral_mean_jsr,
        "gemini_mean_jsr": c.gemini_mean_jsr,
        "pseudo_50_mean_jsr": c.pseudo_50_jsr,
    }
