"""CPU smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.par.benchmarks import TABLE1_T2P, benchmarks_bundle
from ltx_trainer.par.config import PAPER_ARXIV, TRAIN_ITERATIONS
from ltx_trainer.par.datasets import datasets_card
from ltx_trainer.par.paper import framework_card
from ltx_trainer.par.pipeline import ablation_checks, evaluation_demo_run, metrics_demo, train_step


def evaluation_smoke() -> dict[str, Any]:
    par03 = TABLE1_T2P["PAR-0.3B"]
    return {
        "package": "par",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "train_iterations": TRAIN_ITERATIONS,
        "par03_fid": par03["FID"],
        "par03_ds": par03["DS"],
        "beats_panfusion_faed": par03["FAED"] < TABLE1_T2P["PanFusion"]["FAED"],
        "ablation_checks": ablation_checks(),
        "train": train_step(),
        "demo": evaluation_demo_run(),
        "metrics": metrics_demo(),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
        "bundle_keys": list(benchmarks_bundle().keys()),
    }
