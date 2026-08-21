"""Training / evaluation smoke pipeline."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.pano_affordance.benchmarks import table1_ours_hard
from ltx_trainer.pano_affordance.config import LEARNING_RATE, PanoAffordanceConfig
from ltx_trainer.pano_affordance.losses import loss_bce, loss_kl, loss_rtc, total_loss
from ltx_trainer.pano_affordance.pano_affordance_net import PanoAffordanceNet
from ltx_trainer.pano_affordance.synthetic import synthetic_batch


def train_step(
    model: PanoAffordanceNet,
    batch: dict[str, torch.Tensor],
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model.train()
    cfg = model.cfg
    out = model(batch["image"])
    pred = out["a_refined"]
    gt = batch["heatmap_gt"]
    if pred.shape[-1] != gt.shape[-1]:
        pred = torch.nn.functional.interpolate(
            pred, size=gt.shape[-1], mode="linear", align_corners=False
        )
    l_bce = loss_bce(pred, gt)
    l_kl = loss_kl(pred, gt)
    l_rtc = loss_rtc(out["region_feats"], out["text_feats"])
    loss = total_loss(
        l_bce, l_kl, l_rtc,
        lambda_bce=cfg.lambda_bce, lambda_kl=cfg.lambda_kl, lambda_rtc=cfg.lambda_rtc,
    )
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return {
        "loss": float(loss.item()),
        "l_bce": float(l_bce.item()),
        "l_kl": float(l_kl.item()),
        "l_rtc": float(l_rtc.item()),
    }


def evaluation_demo_run(cfg: PanoAffordanceConfig | None = None, *, device: str = "cpu") -> dict[str, Any]:
    cfg = cfg or PanoAffordanceConfig(height=56, width=112)
    dev = torch.device(device)
    batch = synthetic_batch(cfg, batch_size=2, device=dev)
    model = PanoAffordanceNet(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    m = train_step(model, batch, optimizer=opt)
    ref = table1_ours_hard()
    return {
        "device": str(dev),
        "train": m,
        "ref_hard_kld": ref["KLD"],
        "ref_hard_sim": ref["SIM"],
        "ref_hard_nss": ref["NSS"],
    }
