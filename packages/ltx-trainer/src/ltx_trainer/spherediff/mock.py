"""CPU smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.spherediff.benchmarks import TABLE1_USER_STUDY, TABLE2_AUTOMATED, benchmarks_bundle
from ltx_trainer.spherediff.config import PAPER_ARXIV, NUM_SPHERICAL_LATENTS
from ltx_trainer.spherediff.datasets import datasets_card
from ltx_trainer.spherediff.paper import framework_card
from ltx_trainer.spherediff.pipeline import ablation_checks, evaluation_demo_run, train_step


def evaluation_smoke() -> dict[str, Any]:
    static = TABLE1_USER_STUDY["static"]["SphereDiff"]
    auto = TABLE2_AUTOMATED["static"]["SphereDiff"]
    return {
        "package": "spherediff",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "paper_latents": NUM_SPHERICAL_LATENTS,
        "user_distortion_pct": static["distortion"],
        "auto_distortion": auto["distortion"],
        "tuning_free": True,
        "ablation_checks": ablation_checks(),
        "train": train_step(),
        "demo": evaluation_demo_run(),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
        "bundle_keys": list(benchmarks_bundle().keys()),
    }
