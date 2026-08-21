"""End-to-end SEAM pipeline demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.seam.config import SeamConfig
from ltx_trainer.seam.pipeline import evaluation_demo, table1_full_training, table2_shortcut_ablation


def pipeline_demo(*, seed: int = 0, cfg: SeamConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SeamConfig()
    demo = evaluation_demo(seed=seed, cfg=cfg)
    mean_row = next(r for r in table1_full_training() if r["seed"] == "mean")
    baseline = next(r for r in table2_shortcut_ablation() if r["setting"] == "baseline on")
    no_shortcut = next(r for r in table2_shortcut_ablation() if r["setting"] == "noise off + seam off")
    return {
        "demo": demo,
        "ext_auc": mean_row["ext_auc"],
        "ext_auc_std": cfg.ext_auc_std,
        "shortcut_learning_gap": round(baseline["ext_auc"] - no_shortcut["ext_auc"], 4),
        "window_s": cfg.window_s,
    }


def eval_smoke() -> dict[str, Any]:
    out = pipeline_demo(seed=42)
    assert out["ext_auc"] == 0.9713
    assert out["shortcut_learning_gap"] > 0.15
    return {"status": "ok", "ext_auc": out["ext_auc"]}
