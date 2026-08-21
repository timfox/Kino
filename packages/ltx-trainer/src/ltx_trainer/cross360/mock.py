"""CPU smoke for Cross360 stub."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.cross360.benchmarks import benchmarks_bundle
from ltx_trainer.cross360.config import PAPER_ARXIV, PARAMS_M, Cross360Config
from ltx_trainer.cross360.cpfa import CrossProjectionFeatureAlignment
from ltx_trainer.cross360.cross360_net import Cross360NetStub
from ltx_trainer.cross360.datasets import all_datasets_card
from ltx_trainer.cross360.paper import framework_card
from ltx_trainer.cross360.pipeline import evaluation_demo_run, train_step
from ltx_trainer.cross360.synthetic import synthetic_batch
from ltx_trainer.cross360.tangent import sample_tp_patches_from_erp, tp_sampling_layout


def evaluation_smoke() -> dict[str, Any]:
    cfg = Cross360Config(height=64, width=128)
    erp, _ = synthetic_batch(cfg, batch_size=1)
    model = Cross360NetStub(cfg)
    out = model(erp)
    cpfa = CrossProjectionFeatureAlignment(cfg.embed_dim)
    tp = sample_tp_patches_from_erp(erp, patch_size=16)
    f_erp = torch.rand(1, cfg.embed_dim, erp.shape[2], erp.shape[3])
    f_ca = cpfa(f_erp, tp)
    losses = train_step(cfg)
    demo = evaluation_demo_run(cfg)
    bundle = benchmarks_bundle()

    return {
        "package": "cross360",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "params_M": PARAMS_M,
        "depth_shape": list(out["depth"].shape),
        "f_ca_shape": list(f_ca.shape),
        "tp_N": tp_sampling_layout()["N"],
        "L_total": losses["L_total"],
        "demo": demo,
        "m3d_AbsRel": bundle["table1_m3d_ours"]["AbsRel"],
        "struct3d_AbsRel": bundle["table2_struct3d_ours"]["AbsRel"],
        "datasets": all_datasets_card()["datasets"],
        "framework": framework_card()["modules"],
    }
