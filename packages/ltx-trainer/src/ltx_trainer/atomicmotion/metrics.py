"""AMASS evaluation metrics (Sec. 4.1)."""

from __future__ import annotations

import torch
from torch import Tensor


def mpjre(pred_rot: Tensor, gt_rot: Tensor) -> float:
    """Mean per-joint rotation error (degrees proxy from 6D)."""
    diff = torch.abs(pred_rot - gt_rot)
    return float(diff.mean().item() * 30.0)  # scale for demo


def mpjpe(pred_pos: Tensor, gt_pos: Tensor) -> float:
    """Mean per-joint position error (cm proxy)."""
    return float(torch.norm(pred_pos - gt_pos, dim=-1).mean().item() * 100.0)


def mpjve(pred_vel: Tensor, gt_vel: Tensor) -> float:
    return float(torch.norm(pred_vel - gt_vel, dim=-1).mean().item() * 100.0)


def jitter(pred_pos: Tensor) -> float:
    """Smoothness: mean jerk magnitude proxy (10^2 m/s^3 scale)."""
    if pred_pos.shape[0] < 4:
        return 0.0
    vel = pred_pos[1:] - pred_pos[:-1]
    acc = vel[1:] - vel[:-1]
    jerk = acc[1:] - acc[:-1]
    return float(jerk.abs().mean().item() * 1e4)


def evaluate_pose_sequence(
    pred_rot: Tensor,
    gt_rot: Tensor,
    pred_pos: Tensor,
    gt_pos: Tensor,
) -> dict[str, float]:
    pred_vel = pred_pos[1:] - pred_pos[:-1] if pred_pos.shape[0] > 1 else pred_pos
    gt_vel = gt_pos[1:] - gt_pos[:-1] if gt_pos.shape[0] > 1 else gt_pos
    return {
        "mpjre": mpjre(pred_rot, gt_rot),
        "mpjpe": mpjpe(pred_pos, gt_pos),
        "mpjve": mpjve(pred_vel, gt_vel),
        "jitter": jitter(pred_pos),
    }
