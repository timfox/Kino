"""Pairwise + loop closure losses (Sec. III-D)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.clrnet.se3 import (
    compose_transforms,
    identity_transform,
    quat_geodesic_distance,
    transform_from_qt,
    transform_points,
)


def param_distance_loss(
    q_pred: Tensor,
    t_pred: Tensor,
    q_gt: Tensor,
    t_gt: Tensor,
    *,
    lambda_r: float,
    lambda_t: float,
) -> Tensor:
    lq = quat_geodesic_distance(q_pred, q_gt).mean()
    lt = F.smooth_l1_loss(t_pred, t_gt)
    return lambda_r * lq + lambda_t * lt


def point_cloud_distance_loss(
    T_pred: Tensor,
    T_gt: Tensor,
    points: Tensor,
) -> Tensor:
    p_pred = transform_points(T_pred, points)
    p_gt = transform_points(T_gt, points)
    return (p_pred - p_gt).norm(dim=-1).mean()


def pairwise_loss(
    q_pred: Tensor,
    t_pred: Tensor,
    q_gt: Tensor,
    t_gt: Tensor,
    points: Tensor,
    *,
    lambda_pairwise: float,
    lambda_r: float,
    lambda_t: float,
) -> Tensor:
    T_pred = transform_from_qt(q_pred, t_pred)
    T_gt = transform_from_qt(q_gt, t_gt)
    l_param = param_distance_loss(q_pred, t_pred, q_gt, t_gt, lambda_r=lambda_r, lambda_t=lambda_t)
    l_point = point_cloud_distance_loss(T_pred, T_gt, points)
    return (1.0 - lambda_pairwise) * l_param + lambda_pairwise * l_point


def loop_closure_loss(
    q_cl: Tensor,
    t_cl: Tensor,
    q_lr: Tensor,
    t_lr: Tensor,
    q_rc: Tensor,
    t_rc: Tensor,
    *,
    lambda_pairwise: float,
    lambda_r: float,
    lambda_t: float,
) -> Tensor:
    """L_loop on T_loop = T_CL · T_LR · T_RC vs identity (Sec. III-D.2)."""
    T_cl = transform_from_qt(q_cl, t_cl)
    T_lr = transform_from_qt(q_lr, t_lr)
    T_rc = transform_from_qt(q_rc, t_rc)
    T_loop = compose_transforms(T_cl, T_lr, T_rc)
    b = T_loop.shape[0]
    T_id = identity_transform(b, T_loop.device, T_loop.dtype)
    l_mat = F.smooth_l1_loss(T_loop, T_id)
    origin = torch.zeros(b, 64, 3, device=T_loop.device, dtype=T_loop.dtype)
    l_point = point_cloud_distance_loss(T_loop, T_id, origin)
    return (1.0 - lambda_pairwise) * (lambda_r + lambda_t) * 0.5 * l_mat + lambda_pairwise * l_point


def total_calibration_loss(
    l_pairwise: Tensor,
    l_loop: Tensor,
    *,
    lambda_loop: float,
) -> Tensor:
    return (1.0 - lambda_loop) * l_pairwise + lambda_loop * l_loop
