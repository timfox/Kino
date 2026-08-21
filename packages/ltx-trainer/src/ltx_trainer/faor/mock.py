"""CPU smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.faor.benchmarks import TABLE1_ODI_SR, benchmarks_bundle
from ltx_trainer.faor.config import PAPER_ARXIV
from ltx_trainer.faor.datasets import datasets_card
from ltx_trainer.faor.paper import framework_card
from ltx_trainer.faor.pipeline import ablation_table_check, evaluation_demo_run, train_step


def evaluation_smoke() -> dict[str, Any]:
    t8 = TABLE1_ODI_SR["FAOR"][8]
    return {
        "package": "faor",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "ws_psnr_x8": t8[0],
        "ws_ssim_x8": t8[1],
        "ablation_checks": ablation_table_check(),
        "train": train_step(),
        "demo": evaluation_demo_run(2.0),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
        "bundle_keys": list(benchmarks_bundle().keys()),
    }
