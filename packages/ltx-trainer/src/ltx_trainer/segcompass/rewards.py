"""GRPO reward design — format + segmentation (Appendix B.3)."""

from __future__ import annotations

import re

from ltx_trainer.segcompass.matching import hungarian_mean_overlap

_THINKING_BLOCK = re.compile(
    r"<think>\s*(.+?)\s*</think>",
    re.DOTALL,
)
_REF_TOKEN = re.compile(r"<REF>")


def format_score(
    response: str,
    *,
    max_thinking_chars: int = 2048,
) -> float:
    r"""Format reward in {0.0, 0.9, 1.0} per Appendix B.3."""
    blocks = _THINKING_BLOCK.findall(response)
    if len(blocks) != 1:
        return 0.0
    if not blocks[0].strip():
        return 0.0
    ref_matches = list(_REF_TOKEN.finditer(response))
    if len(ref_matches) != 1:
        return 0.0
    think_end = _THINKING_BLOCK.search(response)
    if think_end is None:
        return 0.0
    if ref_matches[0].start() < think_end.end():
        return 0.0
    score = 1.0
    if len(blocks[0]) > max_thinking_chars:
        score = 0.9
    prefix = response[: think_end.start()]
    if prefix.strip():
        score = 0.9
    suffix = response[ref_matches[0].end() :]
    if suffix.strip():
        score = 0.9
    return score


def soft_iou(pred: float, target: float, *, eps: float = 1e-6) -> float:
    """Soft IoU in [0, 1] for single-object mask reward."""
    inter = min(pred, target)
    union = max(pred, target)
    return inter / (union + eps)


def combined_reward(
    format_s: float,
    mask_s: float,
    *,
    w_format: float = 0.3,
    w_seg: float = 0.7,
) -> float:
    """Weighted scalar reward used in GRPO rollouts."""
    return w_format * format_s + w_seg * mask_s


def multi_object_mask_reward(
    pred_masks: list[float],
    gt_masks: list[float],
    *,
    confidence_matched: bool = True,
) -> float:
    """Mean Hungarian overlap + optional binary confidence term."""
    overlap = hungarian_mean_overlap(pred_masks, gt_masks)
    if confidence_matched:
        return overlap
    return 0.5 * overlap
