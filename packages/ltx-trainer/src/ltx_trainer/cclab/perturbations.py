"""Feature- and environment-level adversarial control surfaces."""

from __future__ import annotations

import torch
from torch import Tensor


def perturb_min_rtt(
    rtt_min: Tensor | float,
    factor: Tensor | float,
    *,
    low: float = 0.5,
    high: float = 1.5,
) -> Tensor:
    """Bounded min-RTT perturbation: minRTT · [1-x%, 1+x%] with factor in range."""
    base = torch.as_tensor(rtt_min, dtype=torch.float32)
    f = torch.as_tensor(factor, dtype=torch.float32)
    f = f.clamp(min=low, max=high)
    return base * f


def perturbation_bounds(frac: float) -> tuple[float, float]:
    """Return [1-frac, 1+frac] for adversarial capacity x%."""
    return (1.0 - frac, 1.0 + frac)


def average_absolute_slope(bandwidths: Tensor, window_k: int) -> Tensor:
    """Eq. (8): average |b_i - b_{i-1}| over sliding window of size k."""
    if bandwidths.numel() < 2:
        return torch.tensor(0.0)
    diffs = (bandwidths[1:] - bandwidths[:-1]).abs()
    k = max(1, min(window_k, int(diffs.numel())))
    return diffs[-k:].mean()


def append_bandwidth_step(
    trace: Tensor,
    proposed_mbps: float,
    *,
    delta_budget: float,
    window_k: int,
    bw_min: float = 1.0,
    bw_max: float = 96.0,
) -> tuple[Tensor, float]:
    """Append bandwidth value satisfying smoothness budget St ≤ δ."""
    proposed = max(bw_min, min(bw_max, proposed_mbps))
    if trace.numel() == 0:
        return torch.tensor([proposed]), proposed
    last = float(trace[-1])
    candidate = torch.cat([trace, torch.tensor([proposed])])
    slope = float(average_absolute_slope(candidate, window_k))
    if slope <= delta_budget:
        return candidate, proposed
    # Reduce step toward last value to satisfy budget
    step = delta_budget * window_k
    adjusted = max(bw_min, min(bw_max, last + max(-step, min(step, proposed - last))))
    out = torch.cat([trace, torch.tensor([adjusted])])
    return out, adjusted


def log_scaled_cwnd_smoothness(cwnd: Tensor, window_k: int = 4) -> float:
    """Eq. (9): log-scaled average absolute slope of cwnd."""
    if cwnd.numel() < 2:
        return 0.0
    t = torch.arange(cwnd.numel(), dtype=torch.float32)
    log_c = torch.log(cwnd.clamp(min=1.0))
    slopes = (log_c[1:] - log_c[:-1]).abs() / (t[1:] - t[:-1]).clamp(min=1e-6)
    k = max(1, min(window_k, int(slopes.numel())))
    return float(slopes[-k:].mean())
