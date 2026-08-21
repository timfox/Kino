"""GRPO rewards for TSG and MTR (Sec. 4, A.4.4)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ltx_trainer.mustbench.config import MustBenchConfig


def tsg_grpo_reward(
    pred_t: float,
    gold_t: float,
    audio_duration: float,
    *,
    valid_format: bool = True,
    cfg: MustBenchConfig | None = None,
) -> float:
    """r_TSG = exp(-|t̂ - t*|/15) - 0.5·1_out - 1.0·1_fmt"""
    cfg = cfg or MustBenchConfig()
    scale = cfg.grpo_tsg_scale_s
    out_of_range = pred_t < 0 or pred_t > audio_duration
    fmt_invalid = not valid_format
    base = math.exp(-abs(pred_t - gold_t) / scale)
    penalty = 0.5 * float(out_of_range) + 1.0 * float(fmt_invalid)
    return base - penalty


def _gaussian_smooth_mask(
    intervals: list[tuple[float, float]],
    duration: float,
    *,
    sigma: float = 15.0,
    radius: float = 60.0,
    resolution: int = 512,
) -> np.ndarray:
    t = np.linspace(0, duration, resolution)
    mask = np.zeros(resolution, dtype=np.float64)
    for s, e in intervals:
        center = 0.5 * (s + e)
        width = max(e - s, 1.0)
        for i, ti in enumerate(t):
            if abs(ti - center) <= radius:
                mask[i] = max(mask[i], math.exp(-0.5 * ((ti - center) / sigma) ** 2))
    return mask


def mtr_grpo_reward(
    pred_intervals: list[tuple[float, float]],
    gold_intervals: list[tuple[float, float]],
    audio_duration: float,
    *,
    valid_format: bool = True,
) -> float:
    """Gaussian soft-F1 reward with out-of-range and format penalties."""
    out_of_range = any(s < 0 or e > audio_duration or s >= e for s, e in pred_intervals)
    fmt_invalid = not valid_format
    p_soft = _gaussian_smooth_mask(pred_intervals, audio_duration)
    g_soft = _gaussian_smooth_mask(gold_intervals, audio_duration)
    dot = float(np.dot(p_soft, g_soft))
    denom = float(np.dot(p_soft, p_soft) + np.dot(g_soft, g_soft)) + 1e-8
    soft_f1 = 2 * dot / denom
    penalty = 0.5 * float(out_of_range) + 1.0 * float(fmt_invalid)
    return soft_f1 - penalty


def grpo_smoke(cfg: MustBenchConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MustBenchConfig()
    exact = tsg_grpo_reward(45.0, 45.0, 180.0, cfg=cfg)
    near = tsg_grpo_reward(48.0, 45.0, 180.0, cfg=cfg)
    far = tsg_grpo_reward(90.0, 45.0, 180.0, cfg=cfg)
    oob = tsg_grpo_reward(200.0, 45.0, 180.0, cfg=cfg)
    mtr = mtr_grpo_reward([(70.0, 110.0)], [(72.0, 108.0)], 240.0)
    return {
        "tsg_exact": exact,
        "tsg_near": near,
        "tsg_far": far,
        "tsg_oob": oob,
        "mtr_soft_f1": mtr,
        "ordering": exact > near > far,
    }
