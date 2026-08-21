"""CPU smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.s3po.benchmarks import TABLE1_360VDS_BD, TABLE2_MIG, benchmarks_bundle
from ltx_trainer.s3po.config import PAPER_ARXIV, S3POConfig
from ltx_trainer.s3po.datasets import datasets_card
from ltx_trainer.s3po.paper import framework_card
from ltx_trainer.s3po.pipeline import ablation_modules, evaluation_demo_run, train_step, wss_weights_demo
from ltx_trainer.s3po.s3po_net import S3POStub
from ltx_trainer.s3po.synthetic import synthetic_erp_clip


def evaluation_smoke() -> dict[str, Any]:
    cfg = S3POConfig(lr_height=48, lr_width=64, num_duct_blocks=2)
    model = S3POStub(cfg)
    clip = synthetic_erp_clip(cfg, num_frames=3)
    out = model(clip)
    ours_bd = TABLE1_360VDS_BD["S3PO"]
    rsdn = TABLE1_360VDS_BD["RSDN"]
    ours_mig = TABLE2_MIG["S3PO"]
    bvsr = TABLE2_MIG["BasicVSR"]

    return {
        "package": "s3po",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "vds_bd_psnr": ours_bd["psnr"],
        "beats_rsdn_bd_psnr": ours_bd["psnr"] > rsdn["psnr"],
        "mig_ws_psnr": ours_mig["ws_psnr"],
        "beats_basicvsr_mig": ours_mig["ws_psnr"] > bvsr["ws_psnr"],
        "hr_shape": list(out["hr"].shape),
        "train": train_step(cfg),
        "demo": evaluation_demo_run(cfg),
        "ablation": ablation_modules(cfg),
        "wss_weights": wss_weights_demo(64, 96),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
        "table1_methods": list(TABLE1_360VDS_BD.keys()),
        "bundle_keys": list(benchmarks_bundle().keys()),
    }
