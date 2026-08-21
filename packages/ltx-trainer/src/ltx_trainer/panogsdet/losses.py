"""PanoGSDet objective (Eq. 6–7)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.panogsdet.cubemap import FACE_NAMES


def semantic_cubemap_loss(rendered: dict[str, Tensor], target: dict[str, Tensor]) -> Tensor:
    """L_sem: mean BCE over six cube faces (Eq. 6)."""
    loss = torch.zeros((), device=next(iter(rendered.values())).device)
    for face in FACE_NAMES:
        r = rendered[face]
        g = target[face]
        if r.shape != g.shape:
            g = F.interpolate(g, size=r.shape[-2:], mode="bilinear", align_corners=False)
        loss = loss + F.binary_cross_entropy_with_logits(r, g)
    return loss / len(FACE_NAMES)


def detection_regression_loss(
    pred_boxes: list[Tensor],
    gt_boxes: list[Tensor],
) -> Tensor:
    """TR3D-style L1 on matched foreground boxes (smoke: pairwise min)."""
    device = pred_boxes[0].device if pred_boxes and pred_boxes[0].numel() else torch.device("cpu")
    if not pred_boxes or all(p.numel() == 0 for p in pred_boxes):
        return torch.zeros((), device=device)
    total = torch.zeros((), device=device)
    count = 0
    for pred, gt in zip(pred_boxes, gt_boxes, strict=True):
        if pred.numel() == 0 or gt.numel() == 0:
            continue
        k = min(pred.shape[0], gt.shape[0])
        total = total + F.l1_loss(pred[:k], gt[:k])
        count += 1
    return total / max(count, 1)


def detection_confidence_loss(scores: list[Tensor], gt_scores: list[Tensor]) -> Tensor:
    device = scores[0].device if scores and scores[0].numel() else torch.device("cpu")
    if not scores:
        return torch.zeros((), device=device)
    total = torch.zeros((), device=device)
    count = 0
    for s, t in zip(scores, gt_scores, strict=True):
        if s.numel() == 0:
            continue
        k = min(s.shape[0], t.shape[0])
        total = total + F.binary_cross_entropy(s[:k], t[:k])
        count += 1
    return total / max(count, 1)


def total_loss(
    l_sem: Tensor,
    l_reg: Tensor,
    l_conf: Tensor,
) -> Tensor:
    return l_sem + l_reg + l_conf
