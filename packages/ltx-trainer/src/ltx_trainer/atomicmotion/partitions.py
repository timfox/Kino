"""Atomic Intent Partition (AIP) for SMPL-H 22 joints (Sec. 3.2, Eq. 2)."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

PartitionName = Literal["torso", "left_arm", "right_arm", "left_leg", "right_leg"]

PARTITION_NAMES: tuple[PartitionName, ...] = (
    "torso",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
)

# SMPL-H body joint indices (22 joints, no detailed hand articulation)
TORSO_JOINTS: tuple[int, ...] = (0, 3, 6, 9, 12, 15)  # pelvis, spine chain, neck, head
LEFT_ARM_JOINTS: tuple[int, ...] = (13, 16, 18, 20)
RIGHT_ARM_JOINTS: tuple[int, ...] = (14, 17, 19, 21)
LEFT_LEG_JOINTS: tuple[int, ...] = (1, 4, 7, 10)
RIGHT_LEG_JOINTS: tuple[int, ...] = (2, 5, 8, 11)

PARTITION_JOINTS: dict[PartitionName, tuple[int, ...]] = {
    "torso": TORSO_JOINTS,
    "left_arm": LEFT_ARM_JOINTS,
    "right_arm": RIGHT_ARM_JOINTS,
    "left_leg": LEFT_LEG_JOINTS,
    "right_leg": RIGHT_LEG_JOINTS,
}

# Observable sparse trackers: head + hands (wrist proxies)
OBSERVABLE_JOINTS: tuple[int, ...] = (15, 20, 21)
UNOBSERVABLE_JOINTS: tuple[int, ...] = tuple(
    j for j in range(22) if j not in OBSERVABLE_JOINTS
)


def atomic_intent_partition(x: Tensor) -> dict[PartitionName, Tensor]:
    """AIP(X') -> five functional clusters (Eq. 2).

    x: [T, J, C] or [B, T, J, C]
    """
    squeeze = False
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze = True
    parts: dict[PartitionName, Tensor] = {}
    for name, joints in PARTITION_JOINTS.items():
        parts[name] = x[:, :, joints, :]
    if squeeze:
        return {k: v.squeeze(0) for k, v in parts.items()}
    return parts


def merge_partitions(parts: dict[PartitionName, Tensor]) -> Tensor:
    """Concatenate partitions back to [T, J, C] layout."""
    squeeze = parts["torso"].dim() == 3
    if squeeze:
        b = 1
        t, c = parts["torso"].shape[0], parts["torso"].shape[-1]
    else:
        b = parts["torso"].shape[0]
        t, c = parts["torso"].shape[1], parts["torso"].shape[-1]
    out = torch.zeros(b, t, 22, c, device=parts["torso"].device, dtype=parts["torso"].dtype)
    for name, joints in PARTITION_JOINTS.items():
        p = parts[name]
        if squeeze:
            out[0, :, joints, :] = p
        else:
            out[:, :, joints, :] = p
    return out.squeeze(0) if squeeze else out
