"""LTX DiT inference hooks: perturbations + latent masks."""

from __future__ import annotations

from typing import Any

import torch

from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)

from ltx_trainer.dyna_pruner.config import DynaPrunerConfig
from ltx_trainer.dyna_pruner.env import dyna_pruner_enabled, num_transformer_blocks, read_config
from ltx_trainer.dyna_pruner.torch_ops import (
    block_skip_list_from_importance,
    importance_from_temporal_variance,
    synchronized_masks,
)


def skip_blocks_from_latent(
    latent: torch.Tensor,
    *,
    num_blocks: int | None = None,
    sw: float | None = None,
) -> list[int]:
    cfg = read_config()
    nb = num_blocks if num_blocks is not None else num_transformer_blocks()
    target_sw = sw if sw is not None else cfg.model_sparsity_sw
    if latent.ndim == 5:
        latent = latent[0]
    s = importance_from_temporal_variance(latent.unsqueeze(0))
    return block_skip_list_from_importance(s, num_blocks=nb, sw=target_sw)


def build_perturbation_config(
    skip_blocks: list[int] | None,
    *,
    batch_size: int = 1,
    include_audio: bool = False,
) -> BatchedPerturbationConfig | None:
    if not skip_blocks:
        return None
    perturbations: list[Perturbation] = [
        Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=list(skip_blocks))
    ]
    if include_audio:
        perturbations.append(Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=list(skip_blocks)))
    return BatchedPerturbationConfig(perturbations=[PerturbationConfig(perturbations=perturbations)] * batch_size)


def merge_skip_blocks(*block_lists: list[int] | None) -> list[int]:
    merged: set[int] = set()
    for lst in block_lists:
        if lst:
            merged.update(lst)
    return sorted(merged)


def perturbation_config_for_latent(
    latent: torch.Tensor | None,
    *,
    batch_size: int = 1,
    stg_blocks: list[int] | None = None,
    stg_mode: str | None = None,
    sidecar: dict[str, Any] | None = None,
) -> BatchedPerturbationConfig | None:
    if not dyna_pruner_enabled():
        return None
    dyna_blocks: list[int] | None = None
    if sidecar and isinstance(sidecar.get("dyna_pruner"), dict):
        raw = sidecar["dyna_pruner"].get("skip_blocks")
        if isinstance(raw, list):
            dyna_blocks = [int(x) for x in raw]
    if dyna_blocks is None and latent is not None:
        dyna_blocks = skip_blocks_from_latent(latent)
    skip = merge_skip_blocks(dyna_blocks, stg_blocks)
    include_audio = stg_mode == "stg_av"
    return build_perturbation_config(skip, batch_size=batch_size, include_audio=include_audio)


def apply_latent_data_mask(latent: torch.Tensor, sidecar: dict[str, Any] | None = None) -> torch.Tensor:
    """Scale latent tokens by downsampled data mask when sidecar or live S is available."""
    if not dyna_pruner_enabled():
        return latent
    cfg = read_config()
    if sidecar and isinstance(sidecar.get("dyna_pruner"), dict):
        sparsity = float(sidecar["dyna_pruner"].get("data_sparsity", 0.0))
        keep = max(0.05, 1.0 - sparsity)
        return latent * keep
    if latent.ndim >= 4:
        s = importance_from_temporal_variance(latent.unsqueeze(0) if latent.ndim == 4 else latent)
        sync = synchronized_masks(s, sd=cfg.data_sparsity_sd, sw=cfg.model_sparsity_sw)
        keep = max(0.05, 1.0 - sync["data_sparsity"])
        return latent * keep
    return latent


def metadata_from_latent(latent: torch.Tensor, *, num_blocks: int | None = None) -> dict[str, Any]:
    cfg = read_config()
    nb = num_blocks if num_blocks is not None else num_transformer_blocks()
    if latent.ndim == 5:
        latent = latent[0]
    s = importance_from_temporal_variance(latent.unsqueeze(0) if latent.ndim == 4 else latent.unsqueeze(0))
    sync = synchronized_masks(
        s,
        sd=cfg.data_sparsity_sd,
        sw=cfg.model_sparsity_sw,
        ste_threshold=cfg.ste_threshold,
        kernel=cfg.receptive_field,
        num_blocks=nb,
    )
    return {
        "arxiv_id": cfg.paper_arxiv,
        "data_sparsity": sync["data_sparsity"],
        "model_keep_ratio": sync["model_keep_ratio"],
        "data_threshold": sync["data_threshold"],
        "skip_blocks": sync["skip_blocks"],
        "tau": sync["tau"],
    }
