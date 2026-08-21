"""End-to-end DSFA pipeline demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dsfa.config import DsfaConfig
from ltx_trainer.dsfa.pipeline import evaluation_demo, table2_main_results, table4_dsfa_probability


def pipeline_demo(*, seed: int = 0, cfg: DsfaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DsfaConfig()
    demo = evaluation_demo(seed=seed, cfg=cfg)
    model_k = next(r for r in table2_main_results() if "(k)" in r["model"])
    model_g = next(r for r in table2_main_results() if "(g)" in r["model"])
    best_p = min(table4_dsfa_probability(), key=lambda r: r["cosg_exteval"])
    return {
        "demo": demo,
        "cosg_exteval_eer": model_k["cosg_exteval"],
        "exteval_gain_vs_pt_pct": round(model_g["cosg_exteval"] - model_k["cosg_exteval"], 2),
        "best_dsfa_prob": best_p["probability"],
    }


def eval_smoke() -> dict[str, Any]:
    out = pipeline_demo(seed=42)
    assert out["cosg_exteval_eer"] == 21.80
    assert out["exteval_gain_vs_pt_pct"] > 0.3
    assert out["best_dsfa_prob"] == 0.25
    return {"status": "ok", "cosg_exteval_eer": out["cosg_exteval_eer"]}
