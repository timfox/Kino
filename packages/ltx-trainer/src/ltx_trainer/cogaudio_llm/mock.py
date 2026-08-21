"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cogaudio_llm.config import CogAudioLlmConfig
from ltx_trainer.cogaudio_llm.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table2_empathy_quality,
    table3_emotion_accuracy,
    table4_empathy_ablation,
)


def evaluation_smoke(cfg: CogAudioLlmConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CogAudioLlmConfig()
    demo = evaluation_demo(seed=0, cfg=cfg)
    fw = demo["framework"]
    assert fw["paper"] == "arXiv:2606.06940"
    assert fw["lime"]["utterances"] == 438_884

    ours_t2 = next(r for r in table2_empathy_quality() if r["model"] == "CogAudio-LLM")
    assert ours_t2["human_humdial"] == 3.17
    assert ours_t2["llm_humdial_conf"] == 3.24

    full_t3 = next(r for r in table3_emotion_accuracy() if "Full" in r["model"])
    assert full_t3["overall"] == 49.5
    assert full_t3["conflict"] == 46.0

    full_t4 = next(r for r in table4_empathy_ablation() if "Full" in r["model"])
    assert full_t4["impl_esd_conf"] == 2.90

    b = benchmarks_bundle()
    assert len(b["table2_empathy_quality"]) == 8
    assert len(b["table4_empathy_ablation"]) == 5

    return {
        "status": "ok",
        "paper": fw["paper"],
        "empathy_humdial_conflict": cfg.empathy_humdial_conflict,
        "emo_acc_conflict": cfg.emo_acc_conflict,
    }
