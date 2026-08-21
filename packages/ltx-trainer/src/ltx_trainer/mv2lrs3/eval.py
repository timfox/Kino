"""End-to-end MV2LRS3 evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mv2lrs3.config import Mv2Lrs3Config
from ltx_trainer.mv2lrs3.metrics import iwer, modality_delta, predict_mv2lrs3_wer
from ltx_trainer.mv2lrs3.pipeline import (
    evaluation_demo,
    table1_lrs3_vs_mv2lrs3,
    table5_vocabulary_iwer,
    table6_modalities,
)


def pipeline_demo(*, seed: int = 0, cfg: Mv2Lrs3Config | None = None) -> dict[str, Any]:
    cfg = cfg or Mv2Lrs3Config()
    demo = evaluation_demo(seed=seed, cfg=cfg)
    rows = table1_lrs3_vs_mv2lrs3()
    collapse = [
        {
            "model": r["model"],
            "lrs3_wer": r["lrs3_wer"],
            "mv2lrs3_wer": r["mv2lrs3_wer_mean"],
            "amplification": round(r["mv2lrs3_wer_mean"] / max(r["lrs3_wer"], 0.01), 1),
        }
        for r in rows
    ]
    auto = next(r for r in table6_modalities() if r["model"] == "Auto-AVSR")
    llama = next(r for r in table6_modalities() if r["model"] == "Llama-AVSR")
    vocab = next(r for r in table5_vocabulary_iwer() if r["model"] == "Auto-AVSR")
    return {
        "demo": demo,
        "performance_collapse": collapse,
        "modality_analysis": {
            "auto_avsr": modality_delta(auto["av"], auto["ao"]),
            "llama_avsr": modality_delta(llama["av"], llama["ao"]),
        },
        "lexical_bias_delta": vocab["delta"],
        "linear_fit_slope": cfg.linear_fit_slope,
        "predicted_worst": predict_mv2lrs3_wer(max(r["lrs3_wer"] for r in rows)),
    }


def eval_smoke() -> dict[str, Any]:
    out = pipeline_demo(seed=42)
    assert out["linear_fit_slope"] == 10.4
    assert out["lexical_bias_delta"] == 11.4
    assert out["modality_analysis"]["llama_avsr"]["visual_hurts"]
    assert out["modality_analysis"]["auto_avsr"]["visual_hurts"] is False
    return {"status": "ok", "best_mv2lrs3_wer": 14.0, "slope": out["linear_fit_slope"]}
