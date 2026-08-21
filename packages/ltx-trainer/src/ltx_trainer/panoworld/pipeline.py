"""Training and evaluation pipeline for PanoWorld."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.panoworld.config import PanoWorldConfig
from ltx_trainer.panoworld.losses import total_training_loss
from ltx_trainer.panoworld.panospace_metrics import bf_ov_miou, multiple_choice_accuracy
from ltx_trainer.panoworld.panoworld_net import PanoWorld
from ltx_trainer.panoworld.synthetic import synthetic_batch


def train_step(
    model: PanoWorld,
    batch: dict[str, torch.Tensor],
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model.train()
    out = model(
        batch["rgb"],
        entity_semantics=batch.get("entity_semantics"),
        entity_bfov=batch.get("entity_bfov"),
        entity_depth=batch.get("entity_depth"),
    )
    loss, metrics = total_training_loss(
        out["choice_logits"],
        batch["answers"],
        out["bfov_pred"],
        batch["bfov_gt"],
        batch["bfov_mask"],
    )
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            model.cfg.grad_clip,
        )
        optimizer.step()
    n_graphs = len(out["metadata_graphs"]) if "metadata_graphs" in out else 0
    metrics["num_graphs"] = float(n_graphs)
    return metrics


def evaluate_batch(model: PanoWorld, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        out = model(batch["rgb"])
        pred_choice = out["choice_logits"].argmax(dim=-1)
        acc = multiple_choice_accuracy(pred_choice, batch["answers"])
        miou = 0.0
        if batch["bfov_mask"].any():
            miou = bf_ov_miou(
                out["bfov_pred"][batch["bfov_mask"]],
                batch["bfov_gt"][batch["bfov_mask"]],
            )
    return {"mc_acc": acc, "bfov_miou": miou}


def evaluation_demo_run(
    cfg: PanoWorldConfig | None = None,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    cfg = cfg or PanoWorldConfig(height=64, width=128, hidden_dim=32, patch_size=16)
    dev = torch.device(device)
    batch = synthetic_batch(cfg, batch_size=2, device=dev)
    model = PanoWorld(cfg).to(dev)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train_lr,
        weight_decay=cfg.train_weight_decay,
    )
    m1 = train_step(model, batch, optimizer=opt)
    m2 = evaluate_batch(model, batch)
    return {
        "device": str(dev),
        "train": m1,
        "eval": m2,
        "num_patches": model.encode_panorama(batch["rgb"]).shape[1],
    }
