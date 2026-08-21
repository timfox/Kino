"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ssl_soft_infer.config import SslSoftInferConfig
from ltx_trainer.ssl_soft_infer.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table1_asr_wer,
    table3_phoneme_separability,
)


def evaluation_smoke(cfg: SslSoftInferConfig | None = None) -> dict[str, Any]:
    c = cfg or SslSoftInferConfig()
    demo = evaluation_demo(seed=0, cfg=c)

    wavlm_soft = next(
        r for r in table1_asr_wer() if r["ssl"] == "WavLM" and r["k"] == 4096 and r["infer"] == "soft"
    )
    assert wavlm_soft["erj"] == c.wavlm4096_erj_soft
    assert c.wavlm4096_erj_soft < c.wavlm_cont_erj

    t3 = next(r for r in table3_phoneme_separability() if r["k"] == 4096 and r["task"] == "ASR")
    assert t3["ratio_soft"] == c.wavlm4096_asr_ratio_soft

    assert demo["phoneme"]["hard"]["ratio"] > 0
    assert len(benchmarks_bundle()["table2_synthesis"]) == 3

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "wavlm4096_erj_soft": c.wavlm4096_erj_soft,
        "wavlm4096_vc_spksim_soft": c.wavlm4096_vc_spksim_soft,
        "wavlm4096_asr_ratio_soft": c.wavlm4096_asr_ratio_soft,
    }
