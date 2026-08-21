"""Evaluation metrics from Section IV (LPIPS, temporal smoothness, BRISQUE, trajectory error)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.traj_i2v.benchmarks import TABLE3_QUANTITATIVE


def mse_lpips_proxy(pred: Tensor, target: Tensor) -> float:
    """Cheap perceptual proxy when lpips package unavailable."""
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    p = F.interpolate(pred, size=(256, 256), mode="bilinear", align_corners=False)
    t = F.interpolate(target, size=(256, 256), mode="bilinear", align_corners=False)
    try:
        import lpips  # type: ignore

        loss_fn = lpips.LPIPS(net="alex")
        with torch.no_grad():
            return float(loss_fn(p * 2 - 1, t * 2 - 1).mean())
    except ImportError:
        return float((p - t).pow(2).mean().sqrt())


def temporal_smoothness(flow_magnitude_mean: float) -> float:
    """Mean Farneback flow magnitude (px/frame); see paper Sec. IV."""
    return flow_magnitude_mean


def mean_flow_magnitude(frames: Tensor) -> float:
    """Average optical-flow magnitude between consecutive frames."""
    if frames.dim() == 3:
        frames = frames.unsqueeze(0)
    n = frames.shape[0]
    if n < 2:
        return 0.0
    mags: list[float] = []
    for i in range(n - 1):
        a = frames[i]
        b = frames[i + 1]
        diff = (b - a).abs().mean(dim=0)
        mags.append(float(diff.mean() * 100.0))
    return sum(mags) / len(mags)


def brisque_proxy(frames: Tensor) -> float:
    """No-reference quality proxy; use piq BRISQUE when installed."""
    if frames.dim() == 3:
        frames = frames.unsqueeze(0)
    try:
        from piq import brisque  # type: ignore

        scores = [float(brisque(f.unsqueeze(0), data_range=1.0)) for f in frames]
        return sum(scores) / len(scores)
    except ImportError:
        # High-frequency energy heuristic (lower = smoother / more natural)
        lap = frames[:, :, 1:, :] - frames[:, :, :-1, :]
        return float(20.0 + lap.abs().mean() * 80.0)


def trajectory_error_px(
    tracked: Tensor,
    target: Tensor,
) -> float:
    """Mean L2 distance between tracked centres and GPS-projected GT (px)."""
    if tracked.shape != target.shape:
        raise ValueError(f"shape mismatch {tracked.shape} vs {target.shape}")
    return float((tracked - target).pow(2).sum(-1).sqrt().mean())


def table3_reference(method: str) -> dict[str, float | str]:
    key = method.lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "sg_i2v": "sg_i2v",
        "sg-i2v": "sg_i2v",
        "optical_flow": "optical_flow",
        "opt_flow": "optical_flow",
        "rife": "rife",
        "ground_truth": "ground_truth",
        "gt": "ground_truth",
    }
    return TABLE3_QUANTITATIVE[aliases.get(key, key)]
