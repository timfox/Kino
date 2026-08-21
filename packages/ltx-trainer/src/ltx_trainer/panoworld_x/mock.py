"""CPU smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.panoworld_x.benchmarks import benchmarks_bundle
from ltx_trainer.panoworld_x.config import PANOEXPLORER_VIDEOS, PAPER_ARXIV, PanoWorldXConfig
from ltx_trainer.panoworld_x.datasets import datasets_card
from ltx_trainer.panoworld_x.paper import framework_card
from ltx_trainer.panoworld_x.pipeline import (
    ablation_branches,
    evaluation_demo_run,
    sphere_edge_connectivity,
    train_step,
)
from ltx_trainer.panoworld_x.panoworld_x_net import PanoWorldXStub
from ltx_trainer.panoworld_x.synthetic import synthetic_erp_frame, synthetic_route


def evaluation_smoke() -> dict[str, Any]:
    cfg = PanoWorldXConfig(height=64, width=128, patch_size=16, num_frames=4)
    model = PanoWorldXStub(cfg, num_blocks=1)
    frame = synthetic_erp_frame(cfg)
    route = synthetic_route(cfg)
    out = model(frame, route)
    bundle = benchmarks_bundle()
    step = train_step(cfg)

    return {
        "package": "panoworld_x",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "panoexplorer_videos": PANOEXPLORER_VIDEOS,
        "recon_shape": list(out["reconstruction"].shape),
        "panoworld_x_psnr_table1": bundle["table1_panoramic"]["PanoWorld-X"]["psnr"],
        "panoworld_x_fvd_table1": bundle["table1_panoramic"]["PanoWorld-X"]["fvd"],
        "r_err_perspective": bundle["table1_camera_controllable"]["PanoWorld-X"]["r_err"],
        "beats_genex_fvd": bundle["table1_panoramic"]["PanoWorld-X"]["fvd"]
        < bundle["table1_panoramic"]["GenEX"]["fvd"],
        "train_loss": step["loss"],
        "demo": evaluation_demo_run(cfg),
        "ablation": ablation_branches(cfg),
        "sphere_seam": sphere_edge_connectivity(cfg),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
    }
