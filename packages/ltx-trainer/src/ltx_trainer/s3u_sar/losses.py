"""Physics-guided losses and confidence-gated supervision — Sec. III."""

from __future__ import annotations

import math
from typing import Any


def heatmap_mse_loss(pred: list[list[float]], target: list[list[float]]) -> float:
    """Lmse — Eq. 8 (stub on flat heatmaps)."""
    if not pred or not target:
        return 0.0
    k = len(pred)
    total = 0.0
    for i in range(k):
        pi, ti = pred[i], target[i]
        n = min(len(pi), len(ti))
        total += sum((pi[j] - ti[j]) ** 2 for j in range(n)) / max(n, 1)
    return total / k


def spatial_softmax(values: list[float], temperature: float) -> list[float]:
    """Eq. 10 — temperature-modulated spatial softmax."""
    t = max(temperature, 1e-6)
    exps = [math.exp(v / t) for v in values]
    s = sum(exps)
    return [e / s for e in exps]


def soft_argmax_1d(values: list[float], temperature: float = 1.0) -> float:
    """Eq. 11 expectation coordinate along one axis."""
    w = spatial_softmax(values, temperature)
    return sum(i * w[i] for i in range(len(w)))


def hetero_loss(
    pred_xy: list[tuple[float, float]],
    gt_xy: list[tuple[float, float]],
    lambdas: list[float],
) -> float:
    """Lhetero — Eq. 12."""
    k = len(pred_xy)
    if k == 0:
        return 0.0
    total = 0.0
    for i in range(k):
        dx = abs(pred_xy[i][0] - gt_xy[i][0])
        dy = abs(pred_xy[i][1] - gt_xy[i][1])
        d1 = dx + dy
        lam = lambdas[i] if i < len(lambdas) else 1.0
        total += 1.0 - math.exp(-lam * d1)
    return total / k


def topo_loss(
    pred_xy: list[tuple[float, float]],
    gt_xy: list[tuple[float, float]],
    edges: list[tuple[int, int]],
    gamma: float = 1.0,
) -> float:
    """Ltopo — Eq. 15."""
    if not edges:
        return 0.0
    total = 0.0
    for i, j in edges:
        pd = math.hypot(pred_xy[i][0] - pred_xy[j][0], pred_xy[i][1] - pred_xy[j][1])
        gd = math.hypot(gt_xy[i][0] - gt_xy[j][0], gt_xy[i][1] - gt_xy[j][1])
        total += 1.0 - math.exp(-gamma * (pd - gd) ** 2)
    return total / len(edges)


def entropy_loss(probs: list[float], eps: float = 1e-8) -> float:
    """Lentropy — Eq. 16."""
    return -sum(p * math.log(p + eps) for p in probs if p > 0)


def confidence_gated_total(
    lmse: float,
    lhetero_per_kp: list[float],
    lentropy_per_kp: list[float],
    ltopo_per_edge: list[tuple[float, float, float]],
    *,
    confidences: list[float],
    scale: float = 1.0,
    alpha: float = 0.8,
    beta: float = 0.15,
    mu: float = 0.4,
) -> float:
    """Ltotal — Eq. 17 (stub)."""
    hetero = sum(c * (alpha * lh + beta * le) for c, lh, le in zip(confidences, lhetero_per_kp, lentropy_per_kp))
    topo = sum(ci * cj * lt for ci, cj, lt in ltopo_per_edge)
    return scale * lmse + hetero + mu * topo


def loss_components_card() -> dict[str, Any]:
    return {
        "Lmse": "heatmap Frobenius MSE (Eq. 8)",
        "Lhetero": "scattering-intensity-aware localization with adaptive λi (Eq. 12)",
        "Ltopo": "rigid-body manifold edge distance (Eq. 15)",
        "Lentropy": "Shannon entropy sharpening on spatial softmax (Eq. 16)",
        "Ltotal": "confidence-gated joint supervision Ci=max(Wi) (Eq. 17)",
        "backbone": "HRNet multi-scale fused features (Eq. 6–7)",
    }
