"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.univoice.config import UniVoiceConfig
from ltx_trainer.univoice.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table1_main_results,
    table2_ablation,
)


def evaluation_smoke(cfg: UniVoiceConfig | None = None) -> dict[str, Any]:
    c = cfg or UniVoiceConfig()
    demo = evaluation_demo(seed=0, cfg=c)
    full = next(r for r in table2_ablation(c) if "Full" in r["config"])
    no_fac = next(r for r in table2_ablation(c) if "factorized" in r["config"])
    ours = next(r for r in table1_main_results(c) if "UniVoice" in r["model"])

    assert ours["speech_per"] == 5.26
    assert ours["singing_per"] == 16.22
    assert abs(ours["speech_per"] - c.f5_tts_speech_per) <= 0.1
    assert c.singing_per < c.vevo15_singing_per
    assert c.singing_per < c.soul_x_singer_per
    assert full["speech_per"] < no_fac["speech_per"]
    assert full["singing_per"] < no_fac["singing_per"]
    assert len(c.eval_styles) == 12
    assert c.eval_samples == 900
    assert demo["speech_melody_is_null"] is True
    assert len(benchmarks_bundle(c)["table2_ablation"]) == 5

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "speech_per": c.speech_per,
        "singing_per": c.singing_per,
        "vevo15_singing_per": c.vevo15_singing_per,
        "params_b": c.params_b,
        "eval_styles": len(c.eval_styles),
        "eval_samples": c.eval_samples,
    }
