"""HST training / evaluation smoke."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ltx_trainer.legs_over_arms.benchmarks import table1_best_lower_body
from ltx_trainer.legs_over_arms.config import LEARNING_RATE, LegsOverArmsConfig
from ltx_trainer.legs_over_arms.hst_net import HSTStub
from ltx_trainer.legs_over_arms.metrics import min_ade, min_fde, mlade, nll_pos
from ltx_trainer.legs_over_arms.skeleton import FeatureConfig
from ltx_trainer.legs_over_arms.synthetic import synthetic_batch


def train_step(
    model: HSTStub,
    batch: dict[str, torch.Tensor],
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model.train()
    out = model(batch["past_xy"], batch.get("pose_feat"))
    # Best-mode regression loss (smoke)
    pred = out["pred_modes"]
    gt = batch["future_xy"]
    loss = F.smooth_l1_loss(pred[:, :, 0], gt)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return {
        "loss": float(loss.item()),
        "minADE": min_ade(pred.detach().flatten(0, 1), gt.flatten(0, 1)),
        "minFDE": min_fde(pred.detach().flatten(0, 1), gt.flatten(0, 1)),
        "mLADE": mlade(
            pred.detach().flatten(0, 1),
            gt.flatten(0, 1),
            out["mode_logits"].detach().flatten(0, 1),
        ),
        "NLLpos": nll_pos(
            out["mode_logits"].detach().flatten(0, 1),
            pred.detach().flatten(0, 1),
            gt.flatten(0, 1),
        ),
    }


def evaluation_demo_run(
    cfg: LegsOverArmsConfig | None = None,
    *,
    device: str = "cpu",
    feature_config: FeatureConfig = "K3D_L",
) -> dict[str, Any]:
    cfg = cfg or LegsOverArmsConfig()
    dev = torch.device(device)
    batch = synthetic_batch(cfg, feature_config, batch_size=2, device=dev)
    model = HSTStub(cfg, feature_config=feature_config).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    m = train_step(model, batch, optimizer=opt)
    ref = table1_best_lower_body()
    return {
        "device": str(dev),
        "feature_config": feature_config,
        "train": m,
        "ref_minADE_K3D_L": ref["minADE"],
        "ref_baseline_minADE": 0.39,
    }
