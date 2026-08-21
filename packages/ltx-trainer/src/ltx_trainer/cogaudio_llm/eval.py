"""End-to-end CogAudio-LLM pipeline demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cogaudio_llm.config import CogAudioLlmConfig
from ltx_trainer.cogaudio_llm.pipeline import evaluation_demo, table2_empathy_quality, table3_emotion_accuracy


def pipeline_demo(*, seed: int = 0, cfg: CogAudioLlmConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CogAudioLlmConfig()
    demo = evaluation_demo(seed=seed, cfg=cfg)
    ours = next(r for r in table2_empathy_quality() if r["model"] == "CogAudio-LLM")
    full = next(r for r in table3_emotion_accuracy() if "Full" in r["model"])
    return {
        "demo": demo,
        "empathy_esd_conflict": ours["llm_esd_conf"],
        "conflict_acc_pct": full["conflict"],
        "training_stages": 3,
    }


def eval_smoke() -> dict[str, Any]:
    out = pipeline_demo(seed=42)
    assert out["empathy_esd_conflict"] == 2.90
    assert out["conflict_acc_pct"] == 46.0
    return {"status": "ok", "empathy_esd_conflict": out["empathy_esd_conflict"]}
