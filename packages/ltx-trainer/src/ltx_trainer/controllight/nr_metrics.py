"""No-reference IQA proxies for Light100K interpolation ablation (Table 6)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.controllight.retinex import luminance_y


def niqe_proxy(image: Tensor) -> float:
    """Lower is better: penalize noise + flatness (NIQE trends, not official metric)."""
    y = luminance_y(image.clamp(0, 1))
    if y.dim() == 2:
        y = y.unsqueeze(0).unsqueeze(0)
    elif y.dim() == 3:
        y = y.unsqueeze(0)
    lap = (
        -4 * y
        + F.pad(y, (0, 0, 1, 0))[:, :, 1:, :]
        + F.pad(y, (0, 0, 0, 1))[:, :, :-1, :]
        + F.pad(y, (1, 0, 0, 0))[:, :, :, 1:]
        + F.pad(y, (0, 1, 0, 0))[:, :, :, :-1]
    )
    noise = float(lap.std().item())
    flat = float(1.0 / (y.std().item() + 1e-4))
    return 3.5 + 2.0 * noise + 0.5 * flat


def musiq_proxy(image: Tensor) -> float:
    """Higher is better: sharpness + mid-range brightness (MUSIQ trends)."""
    y = luminance_y(image.clamp(0, 1))
    mean_b = float(y.mean().item())
    gx = y[..., :, 1:] - y[..., :, :-1]
    gy = y[..., 1:, :] - y[..., :-1, :]
    sharp = float((gx.abs().mean() + gy.abs().mean()).item())
    colorful = float(image.std().item())
    return 40.0 + 120.0 * mean_b + 80.0 * sharp + 30.0 * colorful


def interpolation_trajectory_metrics(
    group: dict[float, Tensor],
    *,
    strengths: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
) -> dict[str, list[float]]:
    """NIQE/MUSIQ proxies along pseudo-GT trajectory (Appendix D.2 / Table 6)."""
    niqe: list[float] = []
    musiq: list[float] = []
    for s in strengths:
        key = min(group.keys(), key=lambda k: abs(k - s))
        img = group[key]
        niqe.append(niqe_proxy(img))
        musiq.append(musiq_proxy(img))
    return {"NIQE": niqe, "MUSIQ": musiq}
