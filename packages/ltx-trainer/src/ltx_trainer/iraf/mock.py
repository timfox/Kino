"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.iraf.config import IrafConfig
from ltx_trainer.iraf.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table1_ms_marco_musan,
    table2_instructs2s,
)


def evaluation_smoke(cfg: IrafConfig | None = None) -> dict[str, Any]:
    c = cfg or IrafConfig()
    demo = evaluation_demo(seed=0, cfg=c)

    iraf_ms = next(
        r
        for r in table1_ms_marco_musan()
        if r["method"] == "IRAF" and r["condition"] == "interfering_speakers_only"
    )
    assert iraf_ms["bleu"] == c.msmarco_iraf_bleu
    assert iraf_ms["rsr_pct"] == c.msmarco_iraf_rsr_pct

    iraf_ins = next(
        r
        for r in table2_instructs2s()
        if r["method"] == "IRAF" and r["condition"] == "interfering_speakers_only"
    )
    assert iraf_ins["bleu"] == c.instruct_iraf_bleu
    assert iraf_ins["ssr_pct"] == c.instruct_iraf_ssr_pct

    assert demo["gate"]["suppresses_interference"]
    assert demo["beats_noisyaug_on_msmarco"]
    assert len(benchmarks_bundle()["table1_ms_marco"]) == 6

    rel_ms = (c.msmarco_iraf_bleu - c.msmarco_noisyaug_bleu) / c.msmarco_noisyaug_bleu * 100.0
    assert abs(rel_ms - c.msmarco_iraf_bleu_rel_pct) < 0.1

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "msmarco_iraf_bleu": c.msmarco_iraf_bleu,
        "msmarco_iraf_rsr_pct": c.msmarco_iraf_rsr_pct,
        "instruct_iraf_bleu": c.instruct_iraf_bleu,
        "instruct_iraf_rsr_pct": c.instruct_iraf_rsr_pct,
        "instruct_iraf_ssr_pct": c.instruct_iraf_ssr_pct,
    }
