"""Masked Pose Modeling Guidance (Sec. 3.2, Eq. 1)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.atomicmotion.partitions import OBSERVABLE_JOINTS, UNOBSERVABLE_JOINTS


def build_joint_mask(
    batch_size: int,
    num_joints: int,
    *,
    mask_prob: float = 0.5,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Binary mask m_j: 1 = keep, 0 = zero out."""
    m = torch.ones(batch_size, num_joints)
    for j in UNOBSERVABLE_JOINTS:
        if j >= num_joints:
            continue
        keep = torch.bernoulli(
            torch.full((batch_size,), 1.0 - mask_prob),
            generator=generator,
        )
        m[:, j] = keep
    for j in OBSERVABLE_JOINTS:
        if j < num_joints:
            m[:, j] = 1.0
    return m


def masked_pose_input(
    x_full: Tensor,
    joint_mask: Tensor | None = None,
    *,
    mask_prob: float = 0.5,
) -> Tensor:
    """X'(t,j,c) = X_full(t,j,c) * m_j (Eq. 1)."""
    if x_full.dim() == 3:
        b = 1
        x = x_full.unsqueeze(0)
        squeeze = True
    else:
        b = x_full.shape[0]
        x = x_full
        squeeze = False
    t, j, c = x.shape[1], x.shape[2], x.shape[3]
    if joint_mask is None:
        joint_mask = build_joint_mask(b, j, mask_prob=mask_prob)
    m = joint_mask.to(device=x.device, dtype=x.dtype).view(b, 1, j, 1)
    x_masked = x * m
    return x_masked.squeeze(0) if squeeze else x_masked


def sparse_tracking_input(x_full: Tensor) -> Tensor:
    """Inference: zero-pad unobserved joints, keep head + hands."""
    m = torch.zeros(x_full.shape[-2], device=x_full.device)
    for j in OBSERVABLE_JOINTS:
        if j < m.shape[0]:
            m[j] = 1.0
    if x_full.dim() == 3:
        return x_full * m.view(1, -1, 1)
    return x_full * m.view(1, 1, -1, 1)


def curriculum_mask_ratio(step: int, *, initial: float = 0.8, decay_steps: int = 50_000) -> float:
    """Appendix B: linear decay of masked-training ratio to 0."""
    if step >= decay_steps:
        return 0.0
    return initial * (1.0 - step / decay_steps)
