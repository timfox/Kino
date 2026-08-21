"""CPU smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ob3d.benchmarks import TABLE1_DATASETS, TABLE5_RECON, benchmarks_bundle
from ltx_trainer.ob3d.config import PAPER_ARXIV, NUM_SCENES
from ltx_trainer.ob3d.datasets import datasets_card
from ltx_trainer.ob3d.paper import framework_card
from ltx_trainer.ob3d.pipeline import (
    cpe_demo,
    evaluation_demo_run,
    nvs_demo,
    recon_demo,
    train_step,
    trajectory_demo,
)


def evaluation_smoke() -> dict[str, Any]:
    neus = TABLE5_RECON["NeuS_indoor_ego"]
    colmap = TABLE5_RECON["COLMAP_indoor_ego"]

    return {
        "package": "ob3d",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "num_scenes": NUM_SCENES,
        "only_dataset_with_3d_protocol": TABLE1_DATASETS["OB3D"]["protocol_3d"],
        "neus_better_rmse_than_colmap": neus["RMSE"] < colmap["RMSE"],
        "neus_delta125_high": neus["delta125"] > 0.95,
        "train": train_step(),
        "demo": evaluation_demo_run(),
        "cpe": cpe_demo(),
        "nvs": nvs_demo(),
        "recon": recon_demo(),
        "trajectory": trajectory_demo(),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
        "bundle_keys": list(benchmarks_bundle().keys()),
    }

