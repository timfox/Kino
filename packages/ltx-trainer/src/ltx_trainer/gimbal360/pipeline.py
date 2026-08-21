"""Gimbal360 training / evaluation smoke pipeline."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.gimbal360.benchmarks import table1_ours
from ltx_trainer.gimbal360.config import Gimbal360Config
from ltx_trainer.gimbal360.gimbal360_net import Gimbal360CompletionStub
from ltx_trainer.gimbal360.losses import diffusion_noise_loss, smooth_l1_flow, total_loss
from ltx_trainer.gimbal360.synthetic import synthetic_batch
from ltx_trainer.gimbal360.inference import SamplerConfig, complete_panorama_stub
from ltx_trainer.gimbal360.topology import siamese_shift_loss


def train_step(
    model: Gimbal360CompletionStub,
    batch: dict[str, torch.Tensor],
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model.train()
    cfg = model.cfg
    out = model(batch["perspective"], batch["mask"], batch["z_ref"], shift_delta=4)
    l_ldm = diffusion_noise_loss(out["eps_base"], batch["noise"])
    l_shift = siamese_shift_loss(out["eps_base"], out["eps_shifted"], 4)
    l_flow = smooth_l1_flow(out["flow"], batch["flow_gt"])
    loss = total_loss(l_ldm, l_shift, l_flow, lambda_shift=cfg.lambda_shift, lambda_flow=cfg.lambda_flow)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return {
        "loss": float(loss.item()),
        "l_ldm": float(l_ldm.item()),
        "l_shift": float(l_shift.item()),
        "l_flow": float(l_flow.item()),
    }


def evaluation_demo_run(cfg: Gimbal360Config | None = None, *, device: str = "cpu") -> dict[str, Any]:
    cfg = cfg or Gimbal360Config(erp_height=48, erp_width=96, perspective_height=32, perspective_width=32)
    dev = torch.device(device)
    batch = synthetic_batch(cfg, batch_size=2, device=dev)
    model = Gimbal360CompletionStub(cfg).to(dev)
    from ltx_trainer.gimbal360.config import LEARNING_RATE

    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    m = train_step(model, batch, optimizer=opt)
    ref = table1_ours()
    model.eval()
    infer = complete_panorama_stub(
        model,
        batch["perspective"][:1],
        batch["mask"][:1],
        cfg=cfg,
        sampler=SamplerConfig(num_steps=4, cfg_scale=1.0, step_size=0.1),
    )
    return {
        "device": str(dev),
        "train": m,
        "ref_indoor_fid": ref["indoor"]["FID"],
        "ref_outdoor_fid": ref["outdoor"]["FID"],
        "infer_erp_shape": list(infer["erp"].shape),
        "infer_azimuth_shift": int(infer["total_azimuth_shift"].item()),
    }
