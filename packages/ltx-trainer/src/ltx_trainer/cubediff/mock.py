"""CPU smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cubediff.benchmarks import TABLE1_LAVAL, benchmarks_bundle
from ltx_trainer.cubediff.config import PAPER_ARXIV
from ltx_trainer.cubediff.datasets import datasets_card
from ltx_trainer.cubediff.paper import framework_card
from ltx_trainer.cubediff.pipeline import ablation_table_check, evaluation_demo_run, train_step


def evaluation_smoke() -> dict[str, Any]:
    row = TABLE1_LAVAL["Ours_img_txt"]
    return {
        "package": "cubediff",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "laval_fid": row["fid"],
        "laval_clip_score": row["cs"],
        "ablation_checks": ablation_table_check(),
        "train": train_step(),
        "demo": evaluation_demo_run(),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
        "bundle_keys": list(benchmarks_bundle().keys()),
    }
