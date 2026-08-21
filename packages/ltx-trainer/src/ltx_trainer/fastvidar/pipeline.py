"""Training / eval demo."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from ltx_trainer.fastvidar.complexity import theoretical_speedup
from ltx_trainer.fastvidar.config import FastViDARConfig
from ltx_trainer.fastvidar.erp_fusion import erp_confidence_weighted_fusion, erp_mean_fusion, erp_nearest_fusion
from ltx_trainer.fastvidar.fastvidar_net import FastViDARStub
from ltx_trainer.fastvidar.losses import depth_objective
from ltx_trainer.fastvidar.synthetic import synthetic_depth_gt, synthetic_erp_frames


def evaluation_demo_run(cfg: FastViDARConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FastViDARConfig(height=64, width=128, num_frames=4)
    model = FastViDARStub(cfg)
    frames = synthetic_erp_frames(cfg)
    out = model(frames)
    return {
        "frames_shape": list(frames.shape),
        "per_camera_depth_shape": list(out["per_camera_depth"].shape),
        "fusion_depth_shape": list(out["fusion_depth"].shape),
        "complexity": theoretical_speedup(),
    }


def ablation_global_attention(cfg: FastViDARConfig | None = None) -> dict[str, float]:
    cfg = cfg or FastViDARConfig(height=64, width=128)
    frames = synthetic_erp_frames(cfg)
    full = FastViDARStub(cfg)
    no_g = FastViDARStub(replace(cfg, use_global_attention=False))
    with torch.no_grad():
        df = full(frames)["fusion_depth"].mean()
        dg = no_g(frames)["fusion_depth"].mean()
    return {"full_fusion_mean": float(df), "no_global_fusion_mean": float(dg)}


def fusion_strategy_demo(cfg: FastViDARConfig | None = None) -> dict[str, float]:
    cfg = cfg or FastViDARConfig(height=32, width=64, num_frames=4)
    model = FastViDARStub(cfg)
    with torch.no_grad():
        out = model(synthetic_erp_frames(cfg))
    d = out["per_camera_depth"][0]
    c = out["confidence"][0]
    s, h, w = d.shape
    mask = torch.ones(s, h, w)
    mean_d, _ = erp_mean_fusion(d, mask)
    near_d = erp_nearest_fusion(d, mask)
    w_d = erp_confidence_weighted_fusion(d, mask, c)
    return {
        "mean_depth": float(mean_d.mean()),
        "nearest_depth": float(near_d.mean()),
        "weighted_depth": float(w_d.mean()),
    }


def train_step(cfg: FastViDARConfig | None = None) -> dict[str, float]:
    cfg = cfg or FastViDARConfig(height=64, width=128)
    model = FastViDARStub(cfg)
    frames = synthetic_erp_frames(cfg)
    out = model(frames)
    pred = out["fusion_depth"]
    gt = synthetic_depth_gt(cfg)
    if gt.shape[-2:] != pred.shape[-2:]:
        gt = torch.nn.functional.interpolate(
            gt.unsqueeze(0), size=pred.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(0)
    mask = torch.ones_like(pred)
    losses = depth_objective(pred, gt, mask, lambda_grad=cfg.lambda_grad)
    losses["loss"].backward()
    return {k: float(v.detach()) for k, v in losses.items()}
