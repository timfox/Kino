"""CPU smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.tpgs.benchmarks import TABLE2_RICOH360, benchmarks_bundle
from ltx_trainer.tpgs.config import PAPER_ARXIV
from ltx_trainer.tpgs.datasets import datasets_card
from ltx_trainer.tpgs.paper import framework_card
from ltx_trainer.tpgs.pipeline import (
    ablation_table_check,
    evaluation_demo_run,
    nvs_metrics_demo,
    train_step,
)


def evaluation_smoke() -> dict[str, Any]:
    center = TABLE2_RICOH360["center_10min"]
    return {
        "package": "tpgs",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "center_ours_psnr": center["Ours"]["PSNR"],
        "beats_odgs_center": center["Ours"]["PSNR"] > center["ODGS"]["PSNR"],
        "twelve_views": True,
        "ablation_checks": ablation_table_check(),
        "train": train_step(),
        "demo": evaluation_demo_run(),
        "nvs": nvs_metrics_demo(),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
        "bundle_keys": list(benchmarks_bundle().keys()),
    }
