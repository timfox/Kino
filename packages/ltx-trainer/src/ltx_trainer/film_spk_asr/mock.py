"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.film_spk_asr.config import FilmSpkAsrConfig
from ltx_trainer.film_spk_asr.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table1_asr_wer,
    table2_mcqa,
)


def evaluation_smoke(cfg: FilmSpkAsrConfig | None = None) -> dict[str, Any]:
    c = cfg or FilmSpkAsrConfig()
    demo = evaluation_demo(seed=0, cfg=c)

    spk = next(r for r in table1_asr_wer(c) if r["model"] == "Spk-Cond")
    assert spk["torgo_wer_pp"] == c.spkcond_torgo_wer_pp
    assert spk["neurovoz_wer"] == c.spkcond_neurovoz_wer

    mcqa = next(r for r in table2_mcqa(c) if r["model"] == "Spk-Cond")
    assert mcqa["sex"] == c.mcqa_spkcond_sex
    assert mcqa["sex"] > c.mcqa_base_sex

    base = next(r for r in table1_asr_wer(c) if r["model"] == "Base")
    assert spk["torgo_wer_pp"] < base["torgo_wer"]

    assert demo["normative_embedding_zero"]
    assert demo["identity_preserves_hidden"]
    assert len(benchmarks_bundle()["table2_mcqa"]) == 7

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "spkcond_torgo_wer_pp": c.spkcond_torgo_wer_pp,
        "spkcond_mcqa_sex": c.mcqa_spkcond_sex,
        "trainable_fraction_pct": c.trainable_fraction_pct,
    }
