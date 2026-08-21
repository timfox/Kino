"""PanoLM training / evaluation smoke pipeline."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.panolm.benchmarks import table5_ours
from ltx_trainer.panolm.config import LEARNING_RATE, PanoLMConfig
from ltx_trainer.panolm.losses import contrastive_alignment_loss, vqa_ce_loss
from ltx_trainer.panolm.panolm_net import PanoLMStub
from ltx_trainer.panolm.synthetic import synthetic_batch


def train_step(
    model: PanoLMStub,
    batch: dict[str, torch.Tensor],
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model.train()
    out = model(batch["image"], batch["input_ids"])
    l_ce = vqa_ce_loss(out["logits"], batch["labels"])
    v = out["vision_tokens"].mean(dim=1)
    t = model.text_embed(batch["input_ids"]).mean(dim=1)
    l_rtc = contrastive_alignment_loss(v, t)
    loss = l_ce + 0.1 * l_rtc
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return {
        "loss": float(loss.item()),
        "l_ce": float(l_ce.item()),
        "l_rtc": float(l_rtc.item()),
    }


def evaluation_demo_run(cfg: PanoLMConfig | None = None, *, device: str = "cpu") -> dict[str, Any]:
    cfg = cfg or PanoLMConfig(height=56, width=112)
    dev = torch.device(device)
    batch = synthetic_batch(cfg, batch_size=2, device=dev)
    model = PanoLMStub(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    m = train_step(model, batch, optimizer=opt)
    ref = table5_ours()
    return {
        "device": str(dev),
        "train": m,
        "ref_avg_gpt_score": ref["avg"],
        "ref_1pano_sft_avg": 41.42,
    }
