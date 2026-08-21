"""CPU smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.edm.benchmarks import TABLE1_MATTERPORT3D, benchmarks_bundle
from ltx_trainer.edm.config import PAPER_ARXIV
from ltx_trainer.edm.datasets import datasets_card
from ltx_trainer.edm.paper import framework_card
from ltx_trainer.edm.pipeline import ablation_table_check, evaluation_demo_run, train_step


def evaluation_smoke() -> dict[str, Any]:
    t1 = TABLE1_MATTERPORT3D["EDM"]
    return {
        "package": "edm",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "matterport_auc5": t1["auc_5"],
        "viewport_unaware_dense": True,
        "ablation_checks": ablation_table_check(),
        "train": train_step(),
        "demo": evaluation_demo_run(),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
        "bundle_keys": list(benchmarks_bundle().keys()),
    }
