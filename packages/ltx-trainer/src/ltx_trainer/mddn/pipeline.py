"""MDDN stub train/demo."""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F

from ltx_trainer.mddn.config import MddnConfig
from ltx_trainer.mddn.mddn_net import MDDNStub
from ltx_trainer.mddn.mff import fuse_by_addition
from ltx_trainer.mddn.mdde import MDDE
from ltx_trainer.mddn.synthetic import synthetic_lr_erp
from ltx_trainer.mddn.distortion import erp_distortion_map


def train_step(cfg: MddnConfig | None = None) -> dict[str, float]:
    cfg = cfg or MddnConfig()
    lr = synthetic_lr_erp(cfg)
    hr = F.interpolate(lr, scale_factor=cfg.scale, mode="bilinear", align_corners=False)
    model = MDDNStub(cfg)
    out = model(lr)
    loss = F.l1_loss(out["sr"], hr)
    return {"loss": float(loss.item())}


def fusion_ablation(cfg: MddnConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MddnConfig()
    lr = synthetic_lr_erp(cfg)
    h, w = lr.shape[-2:]
    d = erp_distortion_map(h, w, device=lr.device)
    f = MDDNStub(cfg).shallow(lr)
    branches = MDDE(cfg)(f, d)
    feats = list(branches.values())
    mff_cfg = MddnConfig(**{**cfg.__dict__, "use_mff": True})
    add_cfg = MddnConfig(**{**cfg.__dict__, "use_mff": False})
    sr_mff = MDDNStub(mff_cfg)(lr)["sr"]
    sr_add = MDDNStub(add_cfg)(lr)["sr"]
    return {
        "branch_count": len(feats),
        "add_shape": list(fuse_by_addition(feats).shape),
        "sr_mff_mean": float(sr_mff.detach().mean()),
        "sr_add_mean": float(sr_add.detach().mean()),
    }


def evaluation_demo_run(cfg: MddnConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MddnConfig()
    return {"train": train_step(cfg), "fusion_ablation": fusion_ablation(cfg)}
