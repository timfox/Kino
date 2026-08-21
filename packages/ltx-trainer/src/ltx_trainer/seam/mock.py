"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.seam.config import SeamConfig
from ltx_trainer.seam.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table1_full_training,
    table2_shortcut_ablation,
    table5_quantization,
)


def evaluation_smoke(cfg: SeamConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SeamConfig()
    demo = evaluation_demo(seed=0, cfg=cfg)
    fw = demo["framework"]
    assert fw["paper"] == "arXiv:2606.06837"
    assert fw["window_s"] == 8.0

    mean_row = next(r for r in table1_full_training() if r["seed"] == "mean")
    assert mean_row["ext_auc"] == 0.9713
    assert mean_row["test_auc"] == 0.9766

    baseline = next(r for r in table2_shortcut_ablation() if r["setting"] == "baseline on")
    no_shortcut = next(r for r in table2_shortcut_ablation() if r["setting"] == "noise off + seam off")
    assert baseline["test_auc"] < no_shortcut["test_auc"]
    assert baseline["ext_auc"] > no_shortcut["ext_auc"]

    int4 = next(r for r in table5_quantization() if r["precision"] == "INT4")
    assert int4["vram_mb"] == 41.80

    b = benchmarks_bundle()
    assert len(b["table6_backbone_screening"]) == 5

    return {
        "status": "ok",
        "paper": fw["paper"],
        "ext_auc": cfg.ext_auc,
        "int4_vram_mb": cfg.int4_vram_mb,
    }
