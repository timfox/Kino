"""Multi-task objective (Eq. 3)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.mtpano.config import LAMBDA_AUX, LAMBDA_DEPTH, LAMBDA_NORM, LAMBDA_SEM


def semseg_loss(pred: Tensor, target: Tensor) -> Tensor:
    return F.cross_entropy(pred, target.long(), ignore_index=-1)


def depth_loss(pred: Tensor, target: Tensor) -> Tensor:
    return F.l1_loss(pred, target)


def normal_loss(pred: Tensor, target: Tensor) -> Tensor:
    return 1.0 - F.cosine_similarity(pred, target, dim=1).mean()


def total_loss(
    preds: dict[str, Tensor],
    targets: dict[str, Tensor],
    *,
    lambda_sem: float = LAMBDA_SEM,
    lambda_depth: float = LAMBDA_DEPTH,
    lambda_norm: float = LAMBDA_NORM,
    lambda_aux: float = LAMBDA_AUX,
) -> Tensor:
    l = (
        lambda_sem * semseg_loss(preds["semseg"], targets["semseg"])
        + lambda_depth * depth_loss(preds["depth"], targets["depth"])
        + lambda_norm * normal_loss(preds["normals"], targets["normals"])
    )
    if "grad" in targets:
        l = l + lambda_aux * F.l1_loss(preds["grad"], targets["grad"])
    if "edf" in targets:
        l = l + lambda_aux * F.l1_loss(preds["edf"], targets["edf"])
    if "point_map" in targets:
        l = l + lambda_aux * F.l1_loss(preds["point_map"], targets["point_map"])
    return l
