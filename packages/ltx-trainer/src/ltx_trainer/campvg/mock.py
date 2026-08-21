"""CPU smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.campvg.benchmarks import TABLE1_BASELINES, benchmarks_bundle
from ltx_trainer.campvg.campvg_net import CamPVGStub
from ltx_trainer.campvg.config import PAPER_ARXIV, CamPVGConfig
from ltx_trainer.campvg.datasets import datasets_card
from ltx_trainer.campvg.paper import framework_card
from ltx_trainer.campvg.pipeline import (
    ablation_components,
    epipolar_k_demo,
    evaluation_demo_run,
    plucker_demo,
    train_step,
)
from ltx_trainer.campvg.synthetic import synthetic_erp_video, synthetic_poses


def evaluation_smoke() -> dict[str, Any]:
    cfg = CamPVGConfig(height=64, width=128, num_frames=4, epipolar_k=16)
    model = CamPVGStub(cfg)
    frames = synthetic_erp_video(cfg)
    rots, trans = synthetic_poses(cfg)
    out = model(frames, rots, trans)
    bundle = benchmarks_bundle()
    t1 = TABLE1_BASELINES["CamPVG"]
    mc = TABLE1_BASELINES["MotionCtrl"]

    return {
        "package": "campvg",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "psnr_table1": t1["psnr"],
        "beats_motionctrl_psnr": t1["psnr"] > mc["psnr"],
        "beats_cami2v_fvd": t1["fvd"] < TABLE1_BASELINES["CamI2V"]["fvd"],
        "best_epipolar_k": 250,
        "table3_psnr_k250": bundle["table3_epipolar_k"][250]["psnr"],
        "recon_shape": list(out["frames"].shape),
        "plucker_shape": list(out["plucker"].shape),
        "train": train_step(cfg),
        "demo": evaluation_demo_run(cfg),
        "ablation": ablation_components(cfg),
        "epipolar_k": epipolar_k_demo(250),
        "plucker": plucker_demo(cfg),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
    }
