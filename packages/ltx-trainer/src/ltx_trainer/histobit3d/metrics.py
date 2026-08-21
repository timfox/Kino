"""3D segmentation and realism metrics (Sec. 3.2)."""

from __future__ import annotations

import torch
from torch import Tensor


def dice_3d(pred: Tensor, target: Tensor, *, eps: float = 1e-6) -> float:
    """Foreground voxel-level 3D DICE."""
    pred_b = pred > 0.5
    tgt_b = target > 0.5
    inter = (pred_b & tgt_b).sum().float()
    union = pred_b.sum().float() + tgt_b.sum().float()
    if union <= 0:
        return 1.0 if inter <= 0 else 0.0
    return float((2.0 * inter / (union + eps)).item())


def hausdorff95(pred: Tensor, target: Tensor) -> float:
    """95th percentile Hausdorff distance proxy on 2D slice (μm scale applied externally)."""
    pred_pts = torch.argwhere(pred > 0.5).float()
    tgt_pts = torch.argwhere(target > 0.5).float()
    if pred_pts.numel() == 0 or tgt_pts.numel() == 0:
        return float("inf")
    # subsample for speed
    n = min(200, pred_pts.shape[0], tgt_pts.shape[0])
    pi = pred_pts[torch.linspace(0, pred_pts.shape[0] - 1, n).long()]
    ti = tgt_pts[torch.linspace(0, tgt_pts.shape[0] - 1, n).long()]
    dists = torch.cdist(pi, ti)
    d1 = dists.min(dim=1).values
    d2 = dists.min(dim=0).values
    combined = torch.cat([d1, d2])
    k = max(1, int(0.95 * combined.numel()))
    return float(torch.kthvalue(combined, k).values.item())


def mean_nuclei_volume_um3(
    mask: Tensor,
    *,
    voxel_um3: float = 1.0,
) -> float:
    """Mean per-instance volume (connected components approximated by total foreground)."""
    fg = (mask > 0.5).float()
    return float(fg.sum().item() * voxel_um3)


def evaluate_3d_segmentation(
    pred_mask: Tensor,
    gt_mask: Tensor,
    *,
    voxel_um3: float = 0.5,
) -> dict[str, float]:
    return {
        "dice_3d": dice_3d(pred_mask, gt_mask),
        "hd95": hausdorff95(pred_mask, gt_mask),
        "nuclei_volume_um3": mean_nuclei_volume_um3(pred_mask, voxel_um3=voxel_um3),
    }
