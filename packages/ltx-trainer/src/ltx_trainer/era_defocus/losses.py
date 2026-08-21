"""ErA training losses — Sec. 3.4."""

from __future__ import annotations

import math
from typing import Any


def l1(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(abs(a[i] - b[i]) for i in range(n)) / n


def blur_consistency(pred: list[float], obs: list[float], kernel: list[float]) -> float:
    """‖H ⊗ Xpred − Y‖1 stub (1-D circular conv)."""
    if not kernel:
        return l1(pred, obs)
    ksum = sum(kernel) or 1.0
    blurred = []
    for i, px in enumerate(pred):
        acc = 0.0
        for j, kj in enumerate(kernel):
            acc += kj * pred[(i + j) % len(pred)]
        blurred.append(acc / ksum)
    return l1(blurred, obs)


def era_training_loss(
    pred: list[float],
    gt: list[float],
    obs: list[float],
    kernel: list[float],
    *,
    omega: float = 0.5,
) -> float:
    """L = ω‖Xpred − Xgt‖1 + (1 − ω)‖H ⊗ Xpred − Y‖1."""
    recon = l1(pred, gt)
    consist = blur_consistency(pred, obs, kernel)
    return omega * recon + (1.0 - omega) * consist


def psnr(pred: list[float], gt: list[float], peak: float = 1.0) -> float:
    n = min(len(pred), len(gt))
    if n == 0:
        return 0.0
    mse = sum((pred[i] - gt[i]) ** 2 for i in range(n)) / n
    if mse <= 0:
        return 99.0
    return 10.0 * math.log10((peak * peak) / mse)


def loss_components_card() -> dict[str, Any]:
    return {
        "reconstruction": "ω‖Xpred − Xgt‖1",
        "kernel_consistency": "(1−ω)‖H ⊗ Xpred − Y‖1",
        "error_term": "sparse E in ALM (Eq. 2–3)",
        "unrolling": "K=10 ALM blocks with ResUNet Dϕ/Df",
        "psf_model": "compact kernel basis + per-pixel weights",
    }
