"""VAE prep path: apply data masks before encode."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.dyna_pruner.env import dyna_pruner_enabled, num_transformer_blocks, read_config
from ltx_trainer.dyna_pruner.torch_ops import co_prune_video_batch


def maybe_co_prune_batch(video: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, Any]] | None]:
    if not dyna_pruner_enabled():
        return video, None
    cfg = read_config()
    masked, meta = co_prune_video_batch(video, cfg, num_blocks=num_transformer_blocks())
    return masked, meta


def attach_prep_metadata(latent_data: dict[str, Any], meta_item: dict[str, Any] | None) -> dict[str, Any]:
    if not meta_item:
        return latent_data
    out = dict(latent_data)
    out["dyna_pruner"] = meta_item
    return out
