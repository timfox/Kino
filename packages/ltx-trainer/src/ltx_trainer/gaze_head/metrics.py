"""Evaluation metrics from Sec. 3.2 (plausibility + diversity)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.gaze_head.pose import angular_error_deg


def pearson_correlation(x: Tensor, y: Tensor) -> float:
    """Temporal correlation for one dimension; x, y shape (T,)."""
    if x.numel() < 2:
        return 0.0
    xc = x - x.mean()
    yc = y - y.mean()
    denom = xc.norm() * yc.norm()
    if denom < 1e-8:
        return 0.0
    return float((xc * yc).sum() / denom)


def correlation_pitch_yaw(pred: Tensor, target: Tensor) -> tuple[float, float]:
    """Best-of-batch style correlation on pitch and yaw (Table 1 reports best)."""
    return pearson_correlation(pred[:, 0], target[:, 0]), pearson_correlation(pred[:, 1], target[:, 1])


def average_variance_error(pred: Tensor, target: Tensor) -> float:
    """AVE: |Var(pred) - Var(target)| summed over pitch and yaw (deg² if inputs in rad, scale in caller)."""
    vp = pred.var(dim=0)
    vt = target.var(dim=0)
    return float((vp - vt).abs().sum())


def smoothness_jerk(pose: Tensor) -> float:
    """Third temporal derivative magnitude (Sec. 3.2); lower is smoother."""
    if pose.shape[0] < 4:
        return 0.0
    d1 = pose[1:] - pose[:-1]
    d2 = d1[1:] - d1[:-1]
    d3 = d2[1:] - d2[:-1]
    return float(d3.norm(dim=-1).mean())


def average_pairwise_distance(samples: Tensor) -> float:
    """APD among K generated sequences (DLOW-style, Sec. 3.2).

    samples: (K, T, 2) head poses in radians.
    """
    k = samples.shape[0]
    if k < 2:
        return 0.0
    flat = samples.reshape(k, -1)
    dists: list[float] = []
    for i in range(k):
        for j in range(i + 1, k):
            dists.append(float((flat[i] - flat[j]).norm()))
    return sum(dists) / len(dists)


def evaluate_sequence(
    pred: Tensor,
    target: Tensor,
    *,
    rad_to_deg: bool = True,
) -> dict[str, float]:
    """Single predicted head sequence vs ground truth."""
    scale = 180.0 / 3.141592653589793 if rad_to_deg else 1.0
    ang = float(angular_error_deg(pred.unsqueeze(0), target.unsqueeze(0))[0])
    cp, cy = correlation_pitch_yaw(pred, target)
    ave = average_variance_error(pred * scale, target * scale)
    smooth = smoothness_jerk(pred) * scale
    return {
        "angular_error_deg": ang,
        "correlation_pitch": cp,
        "correlation_yaw": cy,
        "ave_deg2": ave,
        "smoothness": smooth,
    }
