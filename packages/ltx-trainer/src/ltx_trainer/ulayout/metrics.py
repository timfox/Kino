"""2D/3D IoU stubs for layout boundaries (Sec. 4.3)."""

from __future__ import annotations

import torch
from torch import Tensor


def iou_1d(pred: Tensor, target: Tensor, tol: float = 0.05) -> Tensor:
    """Column-wise boundary agreement within tolerance (normalized y)."""
    diff = (pred - target).abs()
    match = (diff < tol).float()
    return match.mean(dim=-1)


def iou_2d_floor_stub(pred_boundary: Tensor, gt_boundary: Tensor) -> Tensor:
    """Proxy 2D IoU from 1D boundary curves."""
    return iou_1d(pred_boundary, gt_boundary)


def iou_3d_room_stub(iou_2d: Tensor, height_ratio: float = 0.95) -> Tensor:
    """Approximate 3D IoU from 2D floor IoU and fixed camera height."""
    return iou_2d * height_ratio
