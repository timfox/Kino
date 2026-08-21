"""Referring segmentation metrics (Appendix B.1)."""

from __future__ import annotations


def cumulative_iou(pred: float, gt: float, *, eps: float = 1e-6) -> float:
    """cIoU — mean IoU over matched instances (scalar toy)."""
    inter = min(pred, gt)
    union = max(pred, gt)
    return inter / (union + eps)


def generalized_iou(pred_masks: list[float], gt_masks: list[float], *, eps: float = 1e-6) -> float:
    """gIoU — Hungarian-matched mean IoU for multi-object (scalar toy)."""
    if not pred_masks and not gt_masks:
        return 1.0
    if not pred_masks or not gt_masks:
        return 0.0
    remaining = list(gt_masks)
    total = 0.0
    matched = 0
    for p in pred_masks:
        if not remaining:
            break
        best = max(cumulative_iou(p, g) for g in remaining)
        best_idx = max(range(len(remaining)), key=lambda i: cumulative_iou(p, remaining[i]))
        total += best
        remaining.pop(best_idx)
        matched += 1
    return total / max(len(pred_masks), len(gt_masks), 1)
