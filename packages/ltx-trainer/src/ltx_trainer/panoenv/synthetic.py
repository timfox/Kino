"""Synthetic GRPO batch for smoke training."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.panoenv.config import PanoEnvConfig
from ltx_trainer.panoenv.rewards import RewardStrategy


def synthetic_grpo_batch(
    cfg: PanoEnvConfig,
    *,
    batch_size: int = 2,
    group_k: int | None = None,
    device: torch.device | str | None = None,
    vocab_size: int = 512,
) -> dict[str, Tensor | list[str] | list[RewardStrategy]]:
    dev = torch.device(device) if device is not None else torch.device("cpu")
    k = group_k or cfg.grpo_k
    b = batch_size * k
    image = torch.rand(batch_size, 3, cfg.height, cfg.width, device=dev)
    image = image.repeat_interleave(k, dim=0)
    prompt_ids = torch.randint(0, vocab_size, (b, 12), device=dev)
    labels = torch.randint(0, vocab_size, (b,), device=dev)
    rewards = torch.rand(b, device=dev)
    strategies: list[RewardStrategy] = ["yes_no", "mcq", "distance", "spatial", "counting"] * (b // 5 + 1)
    strategies = strategies[:b]
    responses = [
        "<Reasoning>stub</Reasoning><Answer>Yes</Answer>",
        "<Reasoning>stub</Reasoning><Answer>No</Answer>",
    ] * (b // 2 + 1)
    responses = responses[:b]
    gts = ["Yes", "No"] * (b // 2 + 1)
    gts = gts[:b]
    return {
        "image": image,
        "prompt_ids": prompt_ids,
        "labels": labels,
        "rewards": rewards,
        "strategies": strategies,
        "responses": responses,
        "ground_truth": gts,
    }
