"""CPU smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.hypevpr.benchmarks import TABLE1_P2E, TABLE2_PITTS, TABLE3_SFXL, benchmarks_bundle
from ltx_trainer.hypevpr.config import PAPER_ARXIV
from ltx_trainer.hypevpr.datasets import datasets_card
from ltx_trainer.hypevpr.paper import framework_card
from ltx_trainer.hypevpr.pipeline import (
    ablation_tables,
    evaluation_demo_run,
    norm_hierarchy_demo,
    retrieval_demo,
    train_step,
)


def evaluation_smoke() -> dict[str, Any]:
    ours_b = TABLE2_PITTS["HypeVPR-B_star"]
    ours_l = TABLE2_PITTS["HypeVPR-L_star"]
    eigen = TABLE2_PITTS["EigenPlace_star"]
    salad = TABLE2_PITTS["SALAD"]
    sfxl = TABLE3_SFXL["HypeVPR-L"]

    return {
        "package": "hypevpr",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "pitts_r1_b": ours_b["r1"],
        "pitts_r1_l": ours_l["r1"],
        "beats_eigenplace_speed": ours_b["time_ms"] < eigen["time_ms"],
        "beats_salad_storage": ours_l["storage_mb"] < salad["storage_mb"],
        "sfxl_r1_l": sfxl["r1"],
        "faster_than_salad_sfxl": sfxl["time_s"] < TABLE3_SFXL["SALAD"]["time_s"],
        "table1_faster_than_orhan": TABLE1_P2E["HypeVPR-B_star_ResNet50"]["time_ms"]
        < TABLE1_P2E["Orhan_star"]["time_ms"],
        "train": train_step(),
        "demo": evaluation_demo_run(),
        "retrieval": retrieval_demo(),
        "norm_hierarchy": norm_hierarchy_demo(),
        "ablation": ablation_tables(),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
        "bundle_keys": list(benchmarks_bundle().keys()),
    }
