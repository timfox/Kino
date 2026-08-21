"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.f3_tokenizer.config import F3TokenizerConfig
from ltx_trainer.f3_tokenizer.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table1_reconstruction,
    table2_probing,
    table3_generation,
)


def evaluation_smoke(cfg: F3TokenizerConfig | None = None) -> dict[str, Any]:
    c = cfg or F3TokenizerConfig()
    demo = evaluation_demo(seed=0, cfg=c)

    recon = next(r for r in table1_reconstruction(c) if r.get("best_recon"))
    assert recon["fd_openl3_audiocaps"] == c.recon_fd_openl3_audiocaps
    assert recon["fd_openl3_audiocaps"] < c.vibevoice_fd_openl3_audiocaps

    fsc = next(r for r in table2_probing(c) if r["dataset"] == "FSC")
    assert fsc["f3"] == c.probe_fsc
    assert fsc["f3"] > fsc["w/o_rq"]
    assert fsc["f3"] > fsc["w/o_llm"]

    tta = next(r for r in table3_generation(c)["tta"] if r.get("best_f3"))
    assert tta["clap"] == c.tta_clap
    assert tta["clap"] > c.ming_tta_clap

    assert demo["norm_noise_beats_vibevoice_fd"]
    assert demo["full_objectives_beat_ablations"]
    assert len(benchmarks_bundle()["table2_probing"]) == 5

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "recon_fd_openl3": c.recon_fd_openl3_audiocaps,
        "probe_librispeech_100h": c.probe_librispeech_100h,
        "tts_seed_en_wer": c.tts_seed_en_wer,
        "tta_clap": c.tta_clap,
    }
