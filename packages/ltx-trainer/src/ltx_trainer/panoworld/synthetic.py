"""Synthetic ERP batches for smoke training."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.panoworld.config import PanoWorldConfig
from ltx_trainer.panoworld.instruction_tasks import batch_to_tensors, sample_instruction_batch


def synthetic_batch(
    cfg: PanoWorldConfig,
    batch_size: int = 2,
    *,
    device: torch.device | str | None = None,
) -> dict[str, Tensor]:
    dev = torch.device(device) if device is not None else None
    rgb = torch.rand(batch_size, 3, cfg.height, cfg.width, device=dev)
    samples = sample_instruction_batch(batch_size, cfg.num_choices, device=dev)
    inst = batch_to_tensors(samples, cfg.num_choices, device=dev)
    n_ent = min(4, cfg.max_entities)
    entity_sem = torch.randn(batch_size, n_ent, 8, device=dev)
    entity_bfov = torch.zeros(batch_size, n_ent, 4, device=dev)
    entity_bfov[..., 0] = torch.linspace(-60, 60, n_ent, device=dev)
    entity_bfov[..., 2] = 25.0
    entity_bfov[..., 3] = 18.0
    entity_depth = torch.rand(batch_size, n_ent, device=dev) * 5.0 + 1.0
    return {
        "rgb": rgb,
        **inst,
        "entity_semantics": entity_sem,
        "entity_bfov": entity_bfov,
        "entity_depth": entity_depth,
    }
