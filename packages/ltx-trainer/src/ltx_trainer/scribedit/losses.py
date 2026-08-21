"""Edit-focused flow-matching objective (Eq. 1–3)."""

from __future__ import annotations

import torch
from torch import Tensor


def velocity_target(z1: Tensor, z0: Tensor) -> Tensor:
    """Rectified-flow target v* = z1 − z0."""
    return z1 - z0


def whole_image_fm_loss(v_pred: Tensor, v_star: Tensor) -> Tensor:
    """Eq. (1): L_whole = E[||v_θ − v*||²]."""
    return (v_pred - v_star).pow(2).mean()


def edit_focused_fm_loss(
    v_pred: Tensor,
    v_star: Tensor,
    mask: Tensor,
) -> Tensor:
    """Eq. (2): L_edit = E[M ⊙ ||v_θ − v*||²], normalized over masked area."""
    sq = (v_pred - v_star).pow(2)
    m = mask
    if m.dim() == 2:
        m = m.unsqueeze(0)
    if m.dim() == 3 and sq.dim() == 4:
        m = m.unsqueeze(1).expand(-1, sq.shape[1], -1, -1)
    elif m.shape != sq.shape:
        m = m.expand_as(sq)
    num = (m * sq).sum()
    den = m.sum().clamp(min=1.0)
    return num / den


def scribedit_training_loss(
    v_pred: Tensor,
    v_star: Tensor,
    edit_mask: Tensor | None = None,
    *,
    edit_lambda: float = 0.1,
    use_edit_focus: bool = True,
) -> Tensor:
    """Eq. (3): L = L_whole + λ L_edit (Stage 1); L_whole only when edit mask absent."""
    l_whole = whole_image_fm_loss(v_pred, v_star)
    if not use_edit_focus or edit_mask is None or edit_lambda <= 0:
        return l_whole
    l_edit = edit_focused_fm_loss(v_pred, v_star, edit_mask)
    return l_whole + edit_lambda * l_edit
