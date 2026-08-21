"""AdaMaG sigma-binned guider factories for HQ two-stage pipelines."""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch

from ltx_core.components.guiders import MultiModalGuiderFactory, MultiModalGuiderParams
from ltx_trainer.adamag.sampling import effective_guidance_scale


def infer_adamag_enabled() -> bool:
    return os.environ.get("GOPEX_INFER_ADAMAG", "").strip().lower() in ("1", "true", "yes")


def _params_with_cfg(base: MultiModalGuiderParams, cfg_scale: float) -> MultiModalGuiderParams:
    return MultiModalGuiderParams(
        cfg_scale=cfg_scale,
        stg_scale=base.stg_scale,
        stg_blocks=base.stg_blocks,
        rescale_scale=base.rescale_scale,
        modality_scale=base.modality_scale,
        skip_step=base.skip_step,
    )


def adamag_multimodal_guider_factory(
    base_params: MultiModalGuiderParams,
    sigmas: Sequence[float] | torch.Tensor,
    *,
    negative_context: torch.Tensor | None = None,
) -> MultiModalGuiderFactory:
    """Build a per-sigma guider factory with AdaMaG-modulated CFG scales."""
    if isinstance(sigmas, torch.Tensor):
        sigma_list = [float(s) for s in sigmas.detach().cpu().tolist()]
    else:
        sigma_list = [float(s) for s in sigmas]

    if len(sigma_list) < 2:
        return MultiModalGuiderFactory.constant(base_params, negative_context=negative_context)

    total = len(sigma_list) - 1
    mapping: dict[float, MultiModalGuiderParams] = {}
    for step_idx, sigma in enumerate(sigma_list[:-1]):
        cfg = effective_guidance_scale(
            base_params.cfg_scale,
            step_index=step_idx,
            total_steps=total,
        )
        mapping[sigma] = _params_with_cfg(base_params, cfg)
    return MultiModalGuiderFactory.from_dict(mapping, negative_context=negative_context)
