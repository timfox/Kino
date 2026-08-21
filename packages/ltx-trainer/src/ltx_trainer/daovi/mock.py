"""CPU smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.daovi.benchmarks import TABLE1_BASELINES, benchmarks_bundle
from ltx_trainer.daovi.config import PAPER_ARXIV, DaoviConfig
from ltx_trainer.daovi.daovi_net import DaoviStub
from ltx_trainer.daovi.datasets import datasets_card
from ltx_trainer.daovi.paper import framework_card
from ltx_trainer.daovi.pipeline import (
    ablation_modules,
    distortion_demo,
    evaluation_demo_run,
    geodesic_demo,
    train_step,
)
from ltx_trainer.daovi.synthetic import synthetic_depth, synthetic_erp_video, synthetic_flow, synthetic_masks


def evaluation_smoke() -> dict[str, Any]:
    cfg = DaoviConfig(height=48, width=96, num_frames=4)
    model = DaoviStub(cfg)
    frames = synthetic_erp_video(cfg)
    masks = synthetic_masks(cfg)
    fwd, bwd = synthetic_flow(cfg)
    depth = synthetic_depth(cfg)
    out = model(frames, masks, fwd, bwd, depth)
    bundle = benchmarks_bundle()
    ours = TABLE1_BASELINES["DAOVI"]
    prop = TABLE1_BASELINES["ProPainter"]

    return {
        "package": "daovi",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "psnr_table1": ours["psnr"],
        "beats_propainter_psnr": ours["psnr"] > prop["psnr"],
        "beats_propainter_vfid": ours["vfid"] < prop["vfid"],
        "recon_shape": list(out["frames"].shape),
        "train": train_step(cfg),
        "demo": evaluation_demo_run(cfg),
        "ablation": ablation_modules(cfg),
        "geodesic": geodesic_demo(cfg.width, cfg.height),
        "distortion": distortion_demo(cfg.height, cfg.width),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
        "table1_methods": list(bundle["table1"].keys()),
    }
