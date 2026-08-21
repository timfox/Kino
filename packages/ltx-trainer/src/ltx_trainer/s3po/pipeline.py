"""Training / ablation demos."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from ltx_trainer.s3po.benchmarks import TABLE5_360_SPECIFIC, TABLE7_PROPAGATION
from ltx_trainer.s3po.config import S3POConfig
from ltx_trainer.s3po.erp_weights import wss_distortion_weights
from ltx_trainer.s3po.losses import wss_l1_loss
from ltx_trainer.s3po.s3po_net import S3POStub
from ltx_trainer.s3po.synthetic import synthetic_erp_clip, synthetic_hr_gt


def evaluation_demo_run(cfg: S3POConfig | None = None) -> dict[str, Any]:
    cfg = cfg or S3POConfig(lr_height=48, lr_width=64, num_duct_blocks=2)
    model = S3POStub(cfg)
    clip = synthetic_erp_clip(cfg, num_frames=3)
    with torch.no_grad():
        out = model(clip)
    return {
        "lr_shape": list(clip.shape),
        "hr_shape": list(out["hr"].shape),
        "scale": int(cfg.scale),
    }


def ablation_modules(cfg: S3POConfig | None = None) -> dict[str, float]:
    cfg = cfg or S3POConfig(lr_height=32, lr_width=48, num_duct_blocks=1)
    clip = synthetic_erp_clip(cfg, num_frames=3)
    full = S3POStub(cfg)
    no_att = S3POStub(replace(cfg, use_attention=False))
    with torch.no_grad():
        a = full(clip)["hr"].mean()
        b = no_att(clip)["hr"].mean()
    return {
        "full_mean": float(a),
        "no_attention_mean": float(b),
        "table5_full_psnr": TABLE5_360_SPECIFIC["S3PO_full"]["psnr"],
        "table5_no_att_psnr": TABLE5_360_SPECIFIC["w/o_attention"]["psnr"],
    }


def wss_weights_demo(height: int = 360, width: int = 480) -> dict[str, Any]:
    psi = wss_distortion_weights(height, width)
    equator = psi[:, :, height // 2, :].mean()
    pole = psi[:, :, 0, :].mean()
    return {
        "equator_weight_gt_pole": float(equator > pole),
        "shape": list(psi.shape),
    }


def train_step(cfg: S3POConfig | None = None) -> dict[str, float]:
    cfg = cfg or S3POConfig(lr_height=32, lr_width=48, num_duct_blocks=1)
    model = S3POStub(cfg)
    clip = synthetic_erp_clip(cfg, num_frames=3)
    gt = synthetic_hr_gt(cfg, num_frames=3)
    out = model(clip)
    loss = wss_l1_loss(out["hr"], gt)
    return {
        "wss_l1": float(loss.detach()),
        "table7_full_psnr": TABLE7_PROPAGATION["S3PO_full"]["psnr"],
        "table7_no_da_psnr": TABLE7_PROPAGATION["w/o_domain_adaptation"]["psnr"],
    }
