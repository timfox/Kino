"""Δ-parameterized intrinsic trajectories (Sec. 3.2)."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

from ltx_trainer.deltacam.config import DeltaCamConfig, IntrinsicRange


def normalize_absolute(value: float, spec: IntrinsicRange) -> float:
    """Map physical quantity to [0, 1] with perceptual log spacing (Eq. 5)."""
    v = float(value)
    lo, hi = spec.m_min, spec.m_max
    if hi <= lo:
        return 0.0
    if spec.log_space:
        if v <= 0 or lo <= 0:
            return 0.0
        return (math.log(v / lo) / math.log(hi / lo))
    return (v - lo) / (hi - lo)


def delta_at_time(
    theta_t: Tensor | float,
    theta_anchor: Tensor | float,
    spec: IntrinsicRange,
) -> Tensor | float:
    """Δθ_t = (θ_t − θ_1) / r with r = physical span (Sec. 3.2)."""
    span = spec.m_max - spec.m_min
    if span <= 0:
        return 0.0
    if isinstance(theta_t, Tensor):
        return (theta_t - theta_anchor) / span
    return (float(theta_t) - float(theta_anchor)) / span


def trajectory_to_delta(
    trajectory: Tensor,
    cfg: DeltaCamConfig | None = None,
    *,
    param_keys: Sequence[str] | None = None,
) -> Tensor:
    """Convert absolute trajectory ``[T, K]`` to Δ w.r.t. frame 0.

    ``trajectory`` columns follow ``param_keys`` order (defaults to all ``cfg.ranges`` keys).
    """
    cfg = cfg or DeltaCamConfig()
    keys = list(param_keys or cfg.ranges.keys())
    if trajectory.dim() != 2 or trajectory.shape[1] != len(keys):
        raise ValueError(f"Expected trajectory [T, {len(keys)}], got {tuple(trajectory.shape)}")
    t_len = trajectory.shape[0]
    out = torch.zeros_like(trajectory)
    for j, key in enumerate(keys):
        spec = cfg.range_for(key)
        anchor = trajectory[0, j]
        for t in range(t_len):
            out[t, j] = delta_at_time(trajectory[t, j], anchor, spec)
    return out


def single_effect_ramp(
    effect: str,
    *,
    start: float,
    end: float,
    num_frames: int,
    cfg: DeltaCamConfig | None = None,
) -> Tensor:
    """Smooth 1D ramp for one intrinsic while others stay at anchor (ablation / demo)."""
    cfg = cfg or DeltaCamConfig()
    if effect not in cfg.ranges:
        raise KeyError(f"Unknown effect {effect!r}")
    keys = list(cfg.ranges.keys())
    traj = torch.zeros(num_frames, len(keys))
    j = keys.index(effect)
    for t in range(num_frames):
        alpha = t / max(num_frames - 1, 1)
        traj[t, j] = start + alpha * (end - start)
        if j > 0:
            traj[t, :j] = traj[0, :j]
        if j + 1 < len(keys):
            traj[t, j + 1 :] = traj[0, j + 1 :]
    return trajectory_to_delta(traj, cfg, param_keys=keys)


def lock_to_anchor_delta(num_frames: int, num_params: int) -> Tensor:
    """All-zero Δ — intrinsic locked to anchor frame (same-view style edit)."""
    return torch.zeros(num_frames, num_params)


def dual_effect_ramp(
    effect_a: str,
    effect_b: str,
    *,
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
    num_frames: int,
    cfg: DeltaCamConfig | None = None,
) -> Tensor:
    """Simultaneous smooth ramps on two intrinsics (Fig. 12 / Table 4 multi-effect row)."""
    cfg = cfg or DeltaCamConfig()
    for e in (effect_a, effect_b):
        if e not in cfg.ranges:
            raise KeyError(f"Unknown effect {e!r}")
    if effect_a == effect_b:
        raise ValueError("effect_a and effect_b must differ")
    keys = list(cfg.ranges.keys())
    ia, ib = keys.index(effect_a), keys.index(effect_b)
    traj = torch.zeros(num_frames, len(keys))
    for t in range(num_frames):
        alpha = t / max(num_frames - 1, 1)
        traj[t, ia] = start_a + alpha * (end_a - start_a)
        traj[t, ib] = start_b + alpha * (end_b - start_b)
        for j in range(len(keys)):
            if j == ia or j == ib:
                continue
            traj[t, j] = traj[0, j]
    return trajectory_to_delta(traj, cfg, param_keys=keys)
