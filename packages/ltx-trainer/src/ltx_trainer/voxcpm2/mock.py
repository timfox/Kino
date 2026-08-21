"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.voxcpm2.config import Voxcpm2Config
from ltx_trainer.voxcpm2.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table3_seed_tts_eval,
    table4_inference_recipes,
    table9_instruct_tts_eval,
)


def evaluation_smoke(cfg: Voxcpm2Config | None = None) -> dict[str, Any]:
    c = cfg or Voxcpm2Config()
    demo = evaluation_demo(seed=0, cfg=c)
    v2 = next(r for r in table3_seed_tts_eval() if r["model"] == "VoxCPM2")
    assert v2["en_wer"] == c.seed_en_wer
    assert v2["zh_sim"] == c.seed_zh_sim

    best_recipe = max(table4_inference_recipes(), key=lambda r: r["en_sim"])
    assert best_recipe["recipe"] == "Reference + Continuation"
    assert best_recipe["en_sim"] == c.seed_ref_cont_en_sim

    vox9 = next(r for r in table9_instruct_tts_eval() if r["model"] == "VoxCPM2")
    assert vox9["rp_en"] == c.instruct_rp_en

    assert demo["audiovae"]["matches_config"]
    assert len(demo["sequence"]["modes"]) == 5

    b = benchmarks_bundle()
    assert len(b["table1_family_config"]) == 3

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "internal_30lang_avg_wer": c.internal_30lang_avg_wer,
        "seed_ref_cont_en_sim": c.seed_ref_cont_en_sim,
        "rtf_nanovllm": c.rtf_nanovllm,
    }
