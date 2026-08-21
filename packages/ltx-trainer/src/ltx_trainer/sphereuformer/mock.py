"""CPU smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.sphereuformer.benchmarks import TABLE4_RANK7, benchmarks_bundle
from ltx_trainer.sphereuformer.config import PAPER_ARXIV
from ltx_trainer.sphereuformer.datasets import datasets_card
from ltx_trainer.sphereuformer.paper import framework_card
from ltx_trainer.sphereuformer.pipeline import ablation_table_check, evaluation_demo_run, train_step


def evaluation_smoke() -> dict[str, Any]:
    row = TABLE4_RANK7["OURS"]["s2d3d"]
    return {
        "package": "sphereuformer",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "s2d3d_mae": row["mae"],
        "s2d3d_d1": row["d1"],
        "s2d3d_seg_miou": TABLE4_RANK7["OURS"]["s2d3d_seg"]["miou"],
        "ablation_checks": ablation_table_check(),
        "train": train_step(),
        "demo": evaluation_demo_run(),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
        "bundle_keys": list(benchmarks_bundle().keys()),
    }
