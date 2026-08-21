"""No-reference IQA proxies for Table 1 comparison (smoke / stub)."""

from __future__ import annotations

import torch
from torch import Tensor


def musiq_proxy(x: Tensor) -> float:
    """Higher is better; rough gradient-energy proxy."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    gx = x[:, :, :, 1:] - x[:, :, :, :-1]
    gy = x[:, :, 1:, :] - x[:, :, :-1, :]
    e = (gx.abs().mean() + gy.abs().mean()) / 2
    return float((50 + 30 * e.clamp(0, 1)).item())


def niqe_proxy(x: Tensor) -> float:
    """Lower is better; variance of local std."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    gray = x.mean(dim=1, keepdim=True)
    k = 5
    pad = k // 2
    patches = gray.unfold(2, k, 1).unfold(3, k, 1)
    std = patches.std(dim=(-1, -2))
    return float(std.var().item() + 3.0)


def metric_bundle(x: Tensor) -> dict[str, float]:
    return {
        "musiq_proxy": musiq_proxy(x),
        "niqe_proxy": niqe_proxy(x),
    }
