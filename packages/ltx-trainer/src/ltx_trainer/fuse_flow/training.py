"""Optional train-time regularizers from fuse_flow fold sidecars."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ltx_trainer.config import AvFoldTrainingConfig


def _scalar(batch: dict[str, Any], field: str) -> Tensor | None:
    from ltx_trainer.av_fold_training import _collated_scalar_field

    return _collated_scalar_field(batch, fold_key="fuse_flow", field=field)


def fuse_flow_geometry_regularizer(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Pull batch mean geometry stability toward ``fuse_flow_geometry_target``."""
    if cfg.lambda_fuse_flow_geometry <= 0:
        return None
    stab = _scalar(batch, "geometry_stability_proxy")
    if stab is None:
        return None
    target = float(cfg.fuse_flow_geometry_target)
    return cfg.lambda_fuse_flow_geometry * ((target - stab.clamp(0.0, 1.0)) ** 2).mean()
