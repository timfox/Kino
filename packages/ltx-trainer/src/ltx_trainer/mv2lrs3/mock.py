"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mv2lrs3.config import Mv2Lrs3Config
from ltx_trainer.mv2lrs3.metrics import predict_mv2lrs3_wer
from ltx_trainer.mv2lrs3.models import model_registry
from ltx_trainer.mv2lrs3.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table1_lrs3_vs_mv2lrs3,
    table2_ten_x_subset,
)


def evaluation_smoke(cfg: Mv2Lrs3Config | None = None) -> dict[str, Any]:
    cfg = cfg or Mv2Lrs3Config()
    demo = evaluation_demo(seed=0, cfg=cfg)
    fw = demo["framework"]
    assert fw["paper"] == "arXiv:2606.07259"
    assert fw["linear_fit"]["slope"] == 10.4

    assert len(model_registry()) == 5
    assert len(table1_lrs3_vs_mv2lrs3()) == 5

    auto = next(r for r in table1_lrs3_vs_mv2lrs3() if r["model"] == "Auto-AVSR")
    llama = next(r for r in table1_lrs3_vs_mv2lrs3() if r["model"] == "Llama-AVSR")
    assert auto["mv2lrs3_rank"] == 1
    assert llama["lrs3_rank"] == 1
    assert auto["mv2lrs3_wer_mean"] == 14.0

    t2 = table2_ten_x_subset()
    assert t2[0]["model"] == "Auto-AVSR"
    assert all(t2[i]["mv2lrs3_rank"] == t2[i]["ten_x_rank"] for i in range(5))

    predicted = predict_mv2lrs3_wer(0.77)
    assert abs(predicted - 16.108) < 0.01

    b = benchmarks_bundle()
    assert len(b["table7_error_rates"]) == 4

    return {
        "status": "ok",
        "paper": fw["paper"],
        "best_mv2lrs3_wer_pct": auto["mv2lrs3_wer_mean"],
        "linear_slope": cfg.linear_fit_slope,
        "models": len(cfg.evaluated_models),
    }
