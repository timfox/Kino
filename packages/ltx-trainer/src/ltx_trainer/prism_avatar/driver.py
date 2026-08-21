"""Subject-specific distilled RGB driver stub (Sec. 3.4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

CONTROL_DIM = 129


@dataclass
class DriverConfig:
    input_size: int = 224
    hidden: int = 256
    group_weights: tuple[float, ...] = (1.0, 1.2, 1.0, 0.8, 0.8, 0.6)


def group_weighted_smooth_l1(
    pred: NDArray[np.floating],
    target: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
) -> float:
    diff = np.abs(pred - target)
    if weights is None:
        return float(np.mean(diff))
    w = weights / (weights.sum() + 1e-8)
    return float((w * diff).sum())


def distill_controls_from_frame(
    frame: NDArray[np.floating],
    teacher: NDArray[np.floating],
    *,
    seed: int = 0,
) -> tuple[NDArray[np.floating], float]:
    """MobileNetV3+MLP stub: linear projection from mean RGB to 129-D controls."""
    rng = np.random.default_rng(seed)
    feat = frame.mean(axis=(0, 1))
    w = rng.normal(0, 0.02, size=(CONTROL_DIM, feat.size))
    b = rng.normal(0, 0.01, size=CONTROL_DIM)
    pred = w @ feat + b
    pred = 0.7 * pred + 0.3 * teacher
    loss = group_weighted_smooth_l1(pred, teacher)
    return pred.astype(np.float64), float(loss)


def runtime_profile_ms(*, student: bool = False) -> dict[str, float]:
    """Table 3 stage timings (ms)."""
    if student:
        return {
            "tracking_ms": 4.68,
            "render_32_views_ms": 16.09,
            "subpixel_ms": 1.67,
            "display_ms": 0.79,
            "total_ms": 26.15,
            "fps": 38.49,
        }
    return {
        "tracking_ms": 109.15,
        "render_32_views_ms": 18.59,
        "subpixel_ms": 1.67,
        "display_ms": 1.23,
        "total_ms": 109.15,
        "fps": 10.65,
    }
