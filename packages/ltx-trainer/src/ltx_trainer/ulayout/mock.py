"""CPU smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ulayout.benchmarks import TABLE3_MATTERPORT_LSUN, benchmarks_bundle
from ltx_trainer.ulayout.config import PAPER_ARXIV
from ltx_trainer.ulayout.datasets import datasets_card
from ltx_trainer.ulayout.paper import framework_card
from ltx_trainer.ulayout.pipeline import ablation_table_check, evaluation_demo_run, train_step


def evaluation_smoke() -> dict[str, Any]:
    t3 = TABLE3_MATTERPORT_LSUN["Ours"]
    return {
        "package": "ulayout",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "table3_pano_2d": t3["pano_2d_iou"],
        "table3_lsun_ceiling": t3["lsun_ceiling"],
        "unified_pano_pp": True,
        "ablation_checks": ablation_table_check(),
        "train": train_step(),
        "demo": evaluation_demo_run(),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
        "bundle_keys": list(benchmarks_bundle().keys()),
    }
