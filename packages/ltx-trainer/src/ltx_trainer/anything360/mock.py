"""CPU smoke for 360Anything stub."""

from __future__ import annotations

from typing import Any

from ltx_trainer.anything360.anything360_model import Anything360Stub
from ltx_trainer.anything360.benchmarks import benchmarks_bundle
from ltx_trainer.anything360.conditioning import sequence_concat_conditioning
from ltx_trainer.anything360.config import PAPER_ARXIV, Anything360Config
from ltx_trainer.anything360.datasets import datasets_card
from ltx_trainer.anything360.paper import framework_card
from ltx_trainer.anything360.pipeline import evaluation_demo_run, train_step
from ltx_trainer.anything360.synthetic import synthetic_pair


def evaluation_smoke() -> dict[str, Any]:
    cfg = Anything360Config(erp_height=64, erp_width=128)
    pers, erp = synthetic_pair(cfg, batch_size=1)
    model = Anything360Stub(cfg)
    out = model(pers, erp)
    bundle = benchmarks_bundle()
    losses = train_step(cfg)
    demo = evaluation_demo_run(cfg)

    z_p = model.encode_perspective(pers)
    z_e = model.encode_erp(erp)
    n_tok = sequence_concat_conditioning(z_p, z_e, patch_size=cfg.patch_size).shape[1]

    return {
        "package": "anything360",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "token_count": n_tok,
        "erp_recon_shape": list(out["erp_recon"].shape),
        "loss": losses["loss"],
        "laval_FID": bundle["table1_laval_ours"]["FID"],
        "real_camera_PSNR": bundle["table2_real_camera"]["PSNR"],
        "cle_image_DS": bundle["table5_cle_ds"]["image"]["CLE"],
        "demo": demo,
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
    }
