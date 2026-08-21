"""Training, inference, and evaluation demo for World Tracing."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ltx_trainer.world_tracing.config import WTConfig
from ltx_trainer.world_tracing.downstream import compose_scene_edit, rasterize_layer_depth, voxelize_point_stack
from ltx_trainer.world_tracing.flow_matching import (
    default_ode_schedule,
    flow_endpoint_loss,
    integrate_flow,
    interpolate_endpoint,
    layer_aware_timestep,
)
from ltx_trainer.world_tracing.intrinsics import fit_pinhole_intrinsics
from ltx_trainer.world_tracing.metrics import mean_absolute_error
from ltx_trainer.world_tracing.representation import invalid_pixel_noise_fill, mix_training_loss_mask
from ltx_trainer.world_tracing.synthetic import synthetic_rgba_batch
from ltx_trainer.world_tracing.wt_dit import WTDiT


def train_step(
    model: WTDiT,
    batch: dict[str, Tensor],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    b_single: bool = False,
) -> dict[str, float]:
    rgba = batch["rgba"]
    x0 = batch["xyz"]
    alpha = batch["alpha"]
    b = rgba.shape[0]
    t = layer_aware_timestep(model.cfg.num_layers, b, phase="mixture", device=rgba.device)
    x1 = torch.randn_like(x0)
    xt = interpolate_endpoint(x0, x1, t.view(b, model.cfg.num_layers, 1, 1, 1))
    xnet = invalid_pixel_noise_fill(xt, alpha)
    pred = model(rgba, xnet, t)
    mask = mix_training_loss_mask(alpha, b_single=b_single, num_layers=model.cfg.num_layers)
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    loss = flow_endpoint_loss(pred, x0, mask, lambda_mono=model.cfg.lambda_mono)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    l0_mae = mean_absolute_error(pred.detach()[:, 0, ..., 2], x0[:, 0, ..., 2])
    return {"loss_fm": float(loss.item()), "l0_depth_mae": l0_mae}


@torch.no_grad()
def infer_multilayer(
    model: WTDiT,
    rgba: Tensor,
    *,
    ode_steps: int | None = None,
) -> Tensor:
    steps = ode_steps or model.cfg.ode_steps
    b = rgba.shape[0]
    x1 = torch.randn(b, model.cfg.num_layers, model.cfg.height, model.cfg.width, 3, device=rgba.device)
    schedule = default_ode_schedule(steps, device=rgba.device)

    def _fn(xt, t_scalar, **_) -> Tensor:
        t = torch.full((b, model.cfg.num_layers), float(t_scalar), device=rgba.device)
        alpha = rgba[:, 3]
        xnet = invalid_pixel_noise_fill(xt, alpha)
        return model(rgba, xnet, t)

    return integrate_flow(_fn, x1, schedule)


def evaluation_demo_run(cfg: WTConfig | None = None, *, device: str = "cpu") -> dict[str, Any]:
    cfg = cfg or WTConfig()
    batch = synthetic_rgba_batch(cfg, batch=1, device=device)
    model = WTDiT(cfg).to(device)
    batch = {k: v.to(device) for k, v in batch.items()}
    train = train_step(model, batch)
    pred = infer_multilayer(model, batch["rgba"], ode_steps=cfg.ode_steps)
    intr = fit_pinhole_intrinsics(pred, height=cfg.height, width=cfg.width, valid=batch["alpha"] > 0.5)
    vox = voxelize_point_stack(pred[0])
    depth_track = rasterize_layer_depth(pred[0], layer=0)
    edit = compose_scene_edit(pred[0], pred[0] * 1.05, batch["alpha"][0])
    return {
        "train": train,
        "intrinsics": intr,
        "voxel_count": vox["count"],
        "depth_track_mean": float(depth_track.mean().item()),
        "edit_changed_fraction": float((edit != pred[0]).float().mean().item()),
        "pred_shape": list(pred.shape),
    }
