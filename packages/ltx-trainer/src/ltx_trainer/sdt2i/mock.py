"""CPU smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.sdt2i.benchmarks import TABLE3_MAIN, benchmarks_bundle
from ltx_trainer.sdt2i.config import PAPER_ARXIV
from ltx_trainer.sdt2i.datasets import datasets_card
from ltx_trainer.sdt2i.paper import framework_card
from ltx_trainer.sdt2i.pipeline import ablation_table_check, evaluation_demo_run, train_step


def evaluation_smoke() -> dict[str, Any]:
    t3 = TABLE3_MAIN["MSTD"]
    return {
        "package": "sdt2i",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "mstd_iou": t3["iou"],
        "mstd_fid": t3["fid"],
        "first_sdt2i": True,
        "ablation_checks": ablation_table_check(),
        "train": train_step(),
        "demo_mstd": evaluation_demo_run("mstd"),
        "demo_mpf": evaluation_demo_run("mpf"),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
        "bundle_keys": list(benchmarks_bundle().keys()),
    }
