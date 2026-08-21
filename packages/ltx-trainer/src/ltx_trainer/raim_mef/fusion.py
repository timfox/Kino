"""Baseline fusion operators (exposure weighting, reference frame)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.raim_mef.synthetic import ev_to_gain


def exposure_weights(stack: Tensor, ev_stops: tuple[float, ...] | list[float]) -> Tensor:
    """Per-frame scalar weights from well-exposedness (centered on 0.5 luminance)."""
    n = stack.shape[0]
    weights = []
    for i in range(n):
        lum = stack[i].mean(dim=0, keepdim=True)
        w = torch.exp(-((lum - 0.5) ** 2) / (2 * 0.08**2))
        weights.append(w * ev_to_gain(ev_stops[i]))
    w_stack = torch.stack(weights, dim=0)
    return w_stack / w_stack.sum(dim=0, keepdim=True).clamp(min=1e-6)


def weighted_fusion(stack: Tensor, ev_stops: tuple[float, ...] | list[float]) -> Tensor:
    w = exposure_weights(stack, ev_stops)
    return (stack * w).sum(dim=0).clamp(0, 1)


def middle_exposure_fusion(stack: Tensor) -> Tensor:
    mid = stack.shape[0] // 2
    return stack[mid].clamp(0, 1)


def select_three_frames(stack: Tensor, ev_stops: tuple[float, ...] | list[float]) -> Tensor:
    """WHU-VIP style: 0 EV + ±2 EV only → [3,3,H,W]."""
    ev_list = list(ev_stops)
    try:
        ref_i = ev_list.index(0.0)
    except ValueError:
        ref_i = len(ev_list) // 2
    targets = {0.0, -2.0, 2.0}
    indices = []
    for t in sorted(targets):
        if t in ev_list:
            indices.append(ev_list.index(t))
        else:
            # nearest EV
            indices.append(min(range(len(ev_list)), key=lambda j: abs(ev_list[j] - t)))
    if ref_i not in indices:
        indices[1] = ref_i
    return stack[indices]
