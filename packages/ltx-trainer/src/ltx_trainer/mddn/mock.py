"""CPU smoke for MDDN stub."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mddn.benchmarks import benchmarks_bundle
from ltx_trainer.mddn.config import PAPER_ARXIV, MddnConfig
from ltx_trainer.mddn.datasets import datasets_card
from ltx_trainer.mddn.distortion import erp_distortion_map
from ltx_trainer.mddn.lowrank import parameter_count_full_vs_lowrank
from ltx_trainer.mddn.mddn_net import MDDNStub
from ltx_trainer.mddn.paper import framework_card
from ltx_trainer.mddn.pipeline import evaluation_demo_run, train_step
from ltx_trainer.mddn.synthetic import synthetic_lr_erp


def evaluation_smoke() -> dict[str, Any]:
    cfg = MddnConfig(erp_height=32, erp_width=64, scale=4)
    lr = synthetic_lr_erp(cfg)
    out = MDDNStub(cfg)(lr)
    bundle = benchmarks_bundle()
    d = erp_distortion_map(cfg.erp_height, cfg.erp_width)
    full, low = parameter_count_full_vs_lowrank(32, 64, rank=cfg.low_rank_rank)

    return {
        "package": "mddn",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "sr_shape": list(out["sr"].shape),
        "distortion_pole_lt_equator": float(d[0, 0, 0, 0]) < float(d[0, 0, cfg.erp_height // 2, 0]),
        "flickr_x4_psnr": bundle["table1_flickr_x4"]["MDDN"]["PSNR"],
        "odi_x4_ws_psnr": bundle["table1_odi_sr_x4"]["MDDN"]["WS-PSNR"],
        "mff_beats_addition": bundle["table5_fusion"]["MFF"] > bundle["table5_fusion"]["Addition"],
        "lowrank_param_reduction": low < full,
        "loss": train_step(cfg)["loss"],
        "demo": evaluation_demo_run(cfg),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
    }
