"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dsfa.config import DsfaConfig
from ltx_trainer.dsfa.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table1_cosg_datasets,
    table2_main_results,
    table4_dsfa_probability,
)


def evaluation_smoke(cfg: DsfaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DsfaConfig()
    demo = evaluation_demo(seed=0, cfg=cfg)
    fw = demo["framework"]
    assert fw["paper"] == "arXiv:2606.07494"
    assert fw["framework"] == "DSFA"

    model_k = next(r for r in table2_main_results() if "(k)" in r["model"])
    model_i = next(r for r in table2_main_results() if "(i)" in r["model"])
    assert model_k["cosg_exteval"] < model_i["cosg_exteval"]
    assert model_k["cosg_exteval"] == 21.80

    ext = next(r for r in table1_cosg_datasets() if r["eval_set"] == "CoSG ExtEval")
    assert ext["spoof_models"] == 40

    best_p = min(table4_dsfa_probability(), key=lambda r: r["cosg_exteval"])
    assert best_p["probability"] == 0.25
    assert best_p["cosg_exteval"] == 22.77

    b = benchmarks_bundle()
    assert len(b["table3_layer_ablation"]) == 5

    return {
        "status": "ok",
        "paper": fw["paper"],
        "cosg_exteval_eer": cfg.cosg_exteval_eer_k,
        "cosg_eval_eer": cfg.cosg_eval_eer_k,
    }
