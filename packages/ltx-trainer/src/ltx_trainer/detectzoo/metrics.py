"""Benchmark metrics (Section 3.2)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def binary_auroc(scores: Tensor, labels: Tensor) -> float:
    """AUROC via rank statistic (CPU-friendly stub)."""
    scores = scores.detach().float().cpu()
    labels = labels.detach().float().cpu()
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    n_pos, n_neg = len(pos), len(neg)
    ranks = torch.zeros_like(scores)
    sorted_idx = torch.argsort(scores)
    ranks[sorted_idx] = torch.arange(1, len(scores) + 1, dtype=torch.float32)
    sum_ranks_pos = ranks[labels > 0.5].sum().item()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(max(0.0, min(1.0, auc)))


def equal_error_rate(scores: Tensor, labels: Tensor) -> float:
    """EER via ROC sweep (linear interpolation stub)."""
    scores = scores.detach().float().cpu()
    labels = labels.detach().float().cpu()
    thresholds = torch.linspace(scores.min().item(), scores.max().item(), 200)
    best = 1.0
    for t in thresholds:
        pred = (scores >= t).float()
        fpr = ((pred == 1) & (labels == 0)).sum().float() / max((labels == 0).sum(), 1)
        fnr = ((pred == 0) & (labels == 1)).sum().float() / max((labels == 1).sum(), 1)
        err = abs(fpr - fnr).item()
        best = min(best, err)
    return float(best)


def accuracy_at_threshold(scores: Tensor, labels: Tensor, threshold: float = 0.5) -> float:
    pred = (scores >= threshold).float()
    return float((pred == labels).float().mean().item())


def metrics_bundle(scores: Tensor, labels: Tensor) -> dict[str, float]:
    return {
        "auroc": binary_auroc(scores, labels),
        "eer": equal_error_rate(scores, labels),
        "accuracy": accuracy_at_threshold(scores, labels),
    }


def toolkit_comparison_table() -> list[dict[str, Any]]:
    """Table 1 — DetectZoo vs prior forensic toolkits."""
    return [
        {"name": "TuringBench", "text": True, "image": False, "audio": False, "detectors": 10, "datasets": 1},
        {"name": "MGTBench", "text": True, "image": False, "audio": False, "detectors": 14, "datasets": 3, "unified_api": True},
        {"name": "DeepfakeBench", "text": False, "image": True, "audio": False, "detectors": 28, "datasets": 8, "unified_api": True},
        {"name": "ASVspoof Baselines", "text": False, "image": False, "audio": True, "detectors": 4, "datasets": 1},
        {
            "name": "DetectZoo",
            "text": True,
            "image": True,
            "audio": True,
            "detectors": 61,
            "datasets": 22,
            "unified_api": True,
            "auto_dl": True,
        },
    ]
