"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.sagnac_phi_otdr.config import SagnacPhiOtdrConfig
from ltx_trainer.sagnac_phi_otdr.metrics import accuracy, confusion_matrix, macro_f1, nar_fnr
from ltx_trainer.sagnac_phi_otdr.pipeline import benchmarks_bundle, evaluation_demo, table2_benchmark


def evaluation_smoke(cfg: SagnacPhiOtdrConfig | None = None) -> dict[str, Any]:
    c = cfg or SagnacPhiOtdrConfig()
    demo = evaluation_demo(seed=0, cfg=c)
    fusion = next(r for r in table2_benchmark(c) if r["method"] == "Fusion CNN")
    branch_b = next(r for r in table2_benchmark(c) if "Branch B" in r["method"])

    assert fusion["accuracy"] == 89.79
    assert fusion["macro_f1"] == 89.83
    assert fusion["nar"] == 5.00
    assert fusion["fnr"] == 0.00
    assert fusion["accuracy"] > branch_b["accuracy"]
    assert c.best_grouping_accuracy > c.default_split_accuracy
    assert len(c.event_classes) == 6
    assert c.total_samples == 15419
    assert demo["grouping_gain_vs_default"] == 27.5

    # metric helpers smoke
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 1, 1, 1, 2, 0])
    m = confusion_matrix(y, p, 3)
    assert accuracy(m) > 0
    assert macro_f1(m) >= 0
    nar, fnr = nar_fnr(m, background_idx=0)
    assert nar >= 0 and fnr >= 0
    assert len(benchmarks_bundle(c)["table2_methods"]) == 5

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "fusion_accuracy": c.fusion_accuracy,
        "fusion_macro_f1": c.fusion_macro_f1,
        "fusion_nar": c.fusion_nar,
        "fusion_fnr": c.fusion_fnr,
        "n_classes": c.n_classes,
        "total_samples": c.total_samples,
    }
