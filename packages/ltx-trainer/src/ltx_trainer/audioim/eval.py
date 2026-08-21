"""End-to-end AudioIM pipeline demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.audioim.config import AudioImConfig
from ltx_trainer.audioim.pipeline import evaluation_demo, table1_v2a_performance, table2_style_similarity


def pipeline_demo(*, seed: int = 0, cfg: AudioImConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AudioImConfig()
    demo = evaluation_demo(seed=seed, cfg=cfg)
    t1 = table1_v2a_performance()
    t2 = table2_style_similarity()
    ours = next(r for r in t1 if "ours" in r["method"])
    prompt_only = next(r for r in t1 if "Prompt Masking" in r["method"])
    return {
        "demo": demo,
        "v2a_improvement_vs_prompt": {
            "kl_panns_delta": round(prompt_only["kl_panns"] - ours["kl_panns"], 2),
            "ib_score_delta": round(ours["ib_score"] - prompt_only["ib_score"], 2),
        },
        "style_ss_mos": next(r for r in t2 if "ours" in r["method"])["ss_mos"],
        "mask_ratio": f"3:5 ({cfg.prompt_seconds}s prompt / {cfg.target_seconds}s target)",
    }


def eval_smoke() -> dict[str, Any]:
    out = pipeline_demo(seed=42)
    assert out["v2a_improvement_vs_prompt"]["kl_panns_delta"] == 0.06
    assert out["style_ss_mos"] == 4.06
    return {"status": "ok", "ss_mos": out["style_ss_mos"]}
