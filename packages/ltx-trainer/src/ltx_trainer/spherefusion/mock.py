"""CPU smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.spherefusion.benchmarks import TABLE1_S2D3D, benchmarks_bundle
from ltx_trainer.spherefusion.config import PAPER_ARXIV
from ltx_trainer.spherefusion.datasets import datasets_card
from ltx_trainer.spherefusion.paper import framework_card
from ltx_trainer.spherefusion.pipeline import ablation_table_check, evaluation_demo_run, train_step


def evaluation_smoke() -> dict[str, Any]:
    sf = TABLE1_S2D3D["SphereFusion"]
    return {
        "package": "spherefusion",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "s2d3d_mre": sf["mre"],
        "s2d3d_d1": sf["d1"],
        "inference_s": sf["time_s"],
        "ablation_checks": ablation_table_check(),
        "train": train_step(),
        "demo": evaluation_demo_run(),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
        "bundle_keys": list(benchmarks_bundle().keys()),
    }
