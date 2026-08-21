"""Synthetic PanoVQA batch for smoke training."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.panolm.config import PanoLMConfig


def synthetic_batch(
    cfg: PanoLMConfig,
    batch_size: int = 2,
    *,
    device: torch.device | str | None = None,
    vocab_size: int = 1000,
) -> dict[str, Tensor]:
    dev = torch.device(device) if device is not None else torch.device("cpu")
    image = torch.rand(batch_size, 3, cfg.height, cfg.width, device=dev)
    input_ids = torch.randint(0, vocab_size, (batch_size, 16), device=dev)
    labels = torch.randint(0, vocab_size, (batch_size,), device=dev)
    return {"image": image, "input_ids": input_ids, "labels": labels}
