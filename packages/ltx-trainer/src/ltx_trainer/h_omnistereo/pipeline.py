"""Training and inference pipeline for H-OmniStereo."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ltx_trainer.h_omnistereo.config import HOmniStereoConfig
from ltx_trainer.h_omnistereo.losses import disparity_loss, normal_training_loss, uncertainty_nll_loss
from ltx_trainer.h_omnistereo.metrics import (
    angular_error_deg,
    bad_pixel_fraction,
    delta_accuracy,
    mean_absolute_error,
)
from ltx_trainer.h_omnistereo.normal_net import HeadingAlignedNormalNet
from ltx_trainer.h_omnistereo.stereo_net import HOmniStereoNet
from ltx_trainer.h_omnistereo.synthetic import synthetic_erp_pair


def train_normal_step(
    model: HeadingAlignedNormalNet,
    batch: dict[str, Tensor],
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    pred = model(batch["top"])
    loss = normal_training_loss(pred, batch["normal_ha"])
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    ang = angular_error_deg(pred.detach(), batch["normal_ha"])
    return {"loss_normal": float(loss.item()), "normal_mae_deg": ang["mae"]}


def train_stereo_step(
    model: HOmniStereoNet,
    batch: dict[str, Tensor],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    refine_iters: int = 3,
) -> dict[str, float]:
    out = model(batch["top"], batch["bottom"], refine_iters=refine_iters)
    disps = out["disparities"]
    assert isinstance(disps, list)
    loss = disparity_loss(disps, batch["disparity"])
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    pred = disps[-1]
    return {
        "loss_disp": float(loss.item()),
        "mae": mean_absolute_error(pred.detach(), batch["disparity"]),
    }


def train_uncertainty_step(
    model: HOmniStereoNet,
    batch: dict[str, Tensor],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    refine_iters: int = 3,
) -> dict[str, float]:
    out = model(batch["top"], batch["bottom"], refine_iters=refine_iters)
    disps = out["disparities"]
    sigmas = out["uncertainties"]
    assert isinstance(disps, list) and isinstance(sigmas, list)
    loss = uncertainty_nll_loss(disps[1:], sigmas, batch["disparity"])
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return {"loss_conf": float(loss.item())}


def predict_disparity(
    model: HOmniStereoNet,
    top: Tensor,
    bottom: Tensor,
    *,
    refine_iters: int | None = None,
) -> Tensor:
    model.eval()
    with torch.no_grad():
        out = model(top, bottom, refine_iters=refine_iters)
    disp = out["disparity"]
    assert isinstance(disp, Tensor)
    return disp


def evaluation_demo_run(
    cfg: HOmniStereoConfig | None = None,
    *,
    device: str = "cpu",
    refine_iters: int = 2,
) -> dict[str, Any]:
    cfg = cfg or HOmniStereoConfig(train_crop_w=64, train_crop_h=64, feature_dim=32, cost_groups=4)
    dev = torch.device(device)
    batch = synthetic_erp_pair(cfg, device=dev)
    normal_net = HeadingAlignedNormalNet(cfg).to(dev)
    stereo = HOmniStereoNet(cfg).to(dev)
    normal_net.train()
    stereo.train()
    n_metrics = train_normal_step(normal_net, batch)
    s_metrics = train_stereo_step(stereo, batch, refine_iters=refine_iters)
    u_metrics = train_uncertainty_step(stereo, batch, refine_iters=refine_iters)
    pred_n = normal_net(batch["top"])
    pred_d = predict_disparity(stereo, batch["top"], batch["bottom"], refine_iters=refine_iters)
    return {
        "device": str(dev),
        "normal": n_metrics,
        "stereo": s_metrics,
        "uncertainty": u_metrics,
        "normal_delta5": delta_accuracy(pred_n.detach(), batch["normal_ha"], 5.0),
        "disparity_bp1": bad_pixel_fraction(pred_d, batch["disparity"], 1.0),
    }
