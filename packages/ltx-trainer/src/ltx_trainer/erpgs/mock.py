"""CPU smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.erpgs.benchmarks import TABLE1_NVS, TABLE2_ABLATION, benchmarks_bundle
from ltx_trainer.erpgs.config import PAPER_ARXIV, OPT_ITERATIONS
from ltx_trainer.erpgs.datasets import datasets_card
from ltx_trainer.erpgs.paper import framework_card
from ltx_trainer.erpgs.pipeline import (
    ablation_table_check,
    evaluation_demo_run,
    nvs_metrics_demo,
    train_step,
)


def evaluation_smoke() -> dict[str, Any]:
    barber = TABLE1_NVS["OmniBlender"]["barbershop"]
    return {
        "package": "erpgs",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "iterations": OPT_ITERATIONS,
        "ours_barbershop_psnr": barber["Ours"]["PSNR"],
        "beats_omnigs": barber["Ours"]["PSNR"] > barber["OmniGS"]["PSNR"],
        "table2_all_psnr": TABLE2_ABLATION["All"]["PSNR"],
        "ablation_checks": ablation_table_check(),
        "train": train_step(),
        "demo": evaluation_demo_run(),
        "nvs": nvs_metrics_demo(),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
        "bundle_keys": list(benchmarks_bundle().keys()),
    }
