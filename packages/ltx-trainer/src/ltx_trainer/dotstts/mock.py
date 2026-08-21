"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dotstts.config import DotsttsConfig
from ltx_trainer.dotstts.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table1_audiovae_reconstruction,
    table2_seed_tts_eval,
    table3_minimax_multilingual_average,
    table5_emergent_tts_selected,
)


def evaluation_smoke(cfg: DotsttsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DotsttsConfig()
    demo = evaluation_demo(seed=0, cfg=cfg)
    fw = demo["framework"]
    assert fw["paper"] == "arXiv:2606.07080"
    assert fw["framework"] == "dots.tts"

    vae = next(r for r in table1_audiovae_reconstruction() if "dots.tts" in r["model"])
    assert vae["wer_pct"] == 4.14
    assert vae["sim"] == 0.969

    soar = next(r for r in table2_seed_tts_eval() if r["model"] == "dots.tts (SOAR)")
    pre = next(r for r in table2_seed_tts_eval() if r["model"] == "dots.tts (Pretrain)")
    seed_baseline = next(r for r in table2_seed_tts_eval() if r["model"] == "Seed-TTS")
    assert soar["avg_sim"] > seed_baseline["avg_sim"]
    assert pre["avg_wer"] <= soar["avg_wer"] + 0.05
    assert soar["zh_sim"] == 81.0

    mm = next(r for r in table3_minimax_multilingual_average() if r["model"] == "dots.tts (SOAR)")
    vox = next(r for r in table3_minimax_multilingual_average() if r["model"] == "VoxCPM 2")
    assert mm["avg_sim"] > vox["avg_sim"]

    emergent = next(r for r in table5_emergent_tts_selected() if r["model"] == "dots.tts (SOAR)")
    assert emergent["syntax"] == 65.7

    assert demo["soar"]["reward_free"] is True
    assert demo["meanflow"]["single_conditional_pass"] is True
    assert len(benchmarks_bundle()["training_stages"]) == 7

    return {
        "status": "ok",
        "paper": fw["paper"],
        "seed_avg_sim": cfg.seed_avg_sim,
        "seed_avg_wer_pct": cfg.seed_avg_wer,
        "ttfp_interleaved_ms": cfg.ttfp_interleaved_ms,
    }
