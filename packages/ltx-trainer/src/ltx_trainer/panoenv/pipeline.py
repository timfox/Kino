"""PanoEnv-RL GRPO smoke pipeline."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.panoenv.benchmarks import table3_ours
from ltx_trainer.panoenv.config import LEARNING_RATE, PanoEnvConfig
from ltx_trainer.panoenv.grpo import group_relative_advantage
from ltx_trainer.panoenv.losses import grpo_step_loss
from ltx_trainer.panoenv.panoenv_net import PanoEnvRLStub
from ltx_trainer.panoenv.rewards import total_reward
from ltx_trainer.panoenv.synthetic import synthetic_grpo_batch


def train_grpo_step(
    model: PanoEnvRLStub,
    batch: dict,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    stage: int = 2,
) -> dict[str, float]:
    model.train()
    out = model(batch["image"], batch["prompt_ids"])
    log_probs = out["log_probs"].gather(1, batch["labels"].unsqueeze(1)).squeeze(1)
    old_log_probs = log_probs.detach()
    # Recompute routed rewards from stub responses
    r_list = [
        total_reward(resp, gt, strat)  # type: ignore[arg-type]
        for resp, gt, strat in zip(batch["responses"], batch["ground_truth"], batch["strategies"])
    ]
    rewards = torch.tensor(r_list, device=log_probs.device, dtype=log_probs.dtype)
    adv = group_relative_advantage(rewards)
    kl = torch.zeros((), device=log_probs.device)
    loss = grpo_step_loss(log_probs, old_log_probs, rewards, kl, beta=model.cfg.kl_beta)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
    return {
        "loss": float(loss.item()),
        "mean_reward": float(rewards.mean().item()),
        "mean_advantage": float(adv.mean().item()),
        "stage": float(stage),
    }


def evaluation_demo_run(cfg: PanoEnvConfig | None = None, *, device: str = "cpu") -> dict[str, Any]:
    cfg = cfg or PanoEnvConfig()
    dev = torch.device(device)
    batch = synthetic_grpo_batch(cfg, batch_size=2, device=dev, vocab_size=64)
    model = PanoEnvRLStub(cfg, vocab_size=64).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    m1 = train_grpo_step(model, batch, optimizer=opt, stage=1)
    m2 = train_grpo_step(model, batch, optimizer=opt, stage=2)
    ref = table3_ours()
    return {
        "device": str(dev),
        "stage1": m1,
        "stage2": m2,
        "ref_total_acc": ref["total"],
        "ref_oe_acc": ref["oe"],
    }
