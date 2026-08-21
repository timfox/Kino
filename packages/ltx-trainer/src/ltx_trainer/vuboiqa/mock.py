"""CPU smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.vuboiqa.benchmarks import TABLE1_OIQA_MAIN, benchmarks_bundle
from ltx_trainer.vuboiqa.config import PAPER_ARXIV
from ltx_trainer.vuboiqa.datasets import datasets_card
from ltx_trainer.vuboiqa.paper import framework_card
from ltx_trainer.vuboiqa.pipeline import (
    ablation_table_check,
    correlation_demo,
    evaluation_demo_run,
    train_step,
)


def evaluation_smoke() -> dict[str, Any]:
    t1 = TABLE1_OIQA_MAIN["VU-BOIQA"]
    return {
        "package": "vuboiqa",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "jufe_srcc": t1["jufe_srcc"],
        "params_m": t1["params_m"],
        "gflops": t1["gflops"],
        "viewport_unaware": True,
        "ablation_checks": ablation_table_check(),
        "train": train_step(),
        "demo": evaluation_demo_run(),
        "correlation": correlation_demo(),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
        "bundle_keys": list(benchmarks_bundle().keys()),
    }
