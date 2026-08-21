"""CLRNet training / evaluation smoke pipeline."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.clrnet.benchmarks import table3_clrnet_median
from ltx_trainer.clrnet.clrnet_net import CLRNet, CRNet
from ltx_trainer.clrnet.config import CLRNetConfig, LEARNING_RATE
from ltx_trainer.clrnet.losses import loop_closure_loss, pairwise_loss, total_calibration_loss
from ltx_trainer.clrnet.synthetic import synthetic_batch


def train_step_clrnet(
    model: CLRNet,
    batch: dict[str, torch.Tensor],
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model.train()
    cfg = model.cfg
    preds = model(
        batch["image"],
        batch["lidar_depth"],
        batch["radar_depth"],
        depth_pred=batch["depth_pred"],
    )
    q_cl, t_cl = preds["CL"]
    q_lr, t_lr = preds["LR"]
    q_rc, t_rc = preds["RC"]

    lp = (
        pairwise_loss(
            q_cl, t_cl, batch["q_cl_gt"], batch["t_cl_gt"], batch["points"],
            lambda_pairwise=cfg.lambda_pairwise, lambda_r=cfg.lambda_r, lambda_t=cfg.lambda_t,
        )
        + pairwise_loss(
            q_lr, t_lr, batch["q_lr_gt"], batch["t_lr_gt"], batch["points"],
            lambda_pairwise=cfg.lambda_pairwise, lambda_r=cfg.lambda_r, lambda_t=cfg.lambda_t,
        )
        + pairwise_loss(
            q_rc, t_rc, batch["q_rc_gt"], batch["t_rc_gt"], batch["points"],
            lambda_pairwise=cfg.lambda_pairwise, lambda_r=cfg.lambda_r, lambda_t=cfg.lambda_t,
        )
    )
    ll = loop_closure_loss(
        q_cl, t_cl, q_lr, t_lr, q_rc, t_rc,
        lambda_pairwise=cfg.lambda_pairwise, lambda_r=cfg.lambda_r, lambda_t=cfg.lambda_t,
    )
    loss = total_calibration_loss(lp, ll, lambda_loop=cfg.lambda_loop)

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return {
        "loss": float(loss.item()),
        "l_pairwise": float(lp.item()),
        "l_loop": float(ll.item()),
    }


def train_step_crnet(
    model: CRNet,
    batch: dict[str, torch.Tensor],
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model.train()
    cfg = model.cfg
    q, t = model(batch["image"], batch["radar_depth"], depth_pred=batch["depth_pred"])
    loss = pairwise_loss(
        q, t, batch["q_rc_gt"], batch["t_rc_gt"], batch["points"],
        lambda_pairwise=cfg.lambda_pairwise, lambda_r=cfg.lambda_r, lambda_t=cfg.lambda_t,
    )
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return {"loss": float(loss.item())}


def evaluation_demo_run(cfg: CLRNetConfig | None = None, *, device: str = "cpu") -> dict[str, Any]:
    cfg = cfg or CLRNetConfig(height=64, width=128)
    dev = torch.device(device)
    batch = synthetic_batch(cfg, batch_size=2, device=dev)
    clr = CLRNet(cfg).to(dev)
    cr = CRNet(cfg).to(dev)
    opt = torch.optim.AdamW(clr.parameters(), lr=LEARNING_RATE)
    m_clr = train_step_clrnet(clr, batch, optimizer=opt)
    m_cr = train_step_crnet(cr, batch)
    ref = table3_clrnet_median()
    return {
        "device": str(dev),
        "clrnet": m_clr,
        "crnet": m_cr,
        "ref_cr_transl_median_cm": ref["cr_transl_median"],
        "ref_cl_transl_median_cm": ref["cl_transl_median"],
    }
