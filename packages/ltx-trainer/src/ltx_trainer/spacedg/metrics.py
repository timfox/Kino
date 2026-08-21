"""Evaluation metrics (Sec. 4.1, 4.3)."""

from __future__ import annotations

import re

import torch


def accuracy(pred: str, gt: str) -> float:
    p = _normalize(pred)
    g = _normalize(gt)
    if p == g:
        return 1.0
    if len(g) == 1 and g.isalpha() and p.startswith(g):
        return 1.0
    return 0.0


def mean_relative_accuracy(
    pred: float,
    gt: float,
    *,
    thresholds: tuple[float, ...] | None = None,
) -> float:
    """MRA with Θ = {0.50, …, 0.95} (VSI-Bench style)."""
    if gt <= 0:
        return 1.0 if pred <= 0 else 0.0
    rel = abs(pred - gt) / abs(gt)
    thetas = thresholds or tuple(0.5 + 0.05 * i for i in range(10))
    return sum(1.0 for t in thetas if rel < (1.0 - t)) / len(thetas)


def list_numeric_score(pred: str, gt: str) -> float:
    pf = _parse_float_list(pred)
    gf = _parse_float_list(gt)
    if not pf or not gf or len(pf) != len(gf):
        return 0.0
    return sum(mean_relative_accuracy(a, b) for a, b in zip(pf, gf, strict=True)) / len(pf)


def point_biserial_abs(clean_scores: list[float], degraded_scores: list[float]) -> float:
    x = torch.tensor([0.0] * len(clean_scores) + [1.0] * len(degraded_scores))
    y = torch.tensor(clean_scores + degraded_scores, dtype=torch.float32)
    if y.std() < 1e-8 or x.std() < 1e-8:
        return 0.0
    r = torch.corrcoef(torch.stack([x, y]))[0, 1].item()
    return abs(float(r))


def degradation_recognition_accuracy(pred_label: str, gt_label: str) -> float:
    return 1.0 if _normalize(pred_label).replace(" ", "_") == _normalize(gt_label).replace(" ", "_") else 0.0


def _normalize(s: str) -> str:
    s = s.strip().lower()
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", s, re.I | re.S)
    if m:
        s = m.group(1)
    return re.sub(r"[^a-z0-9._-]+", "", s)


def _parse_float_list(s: str) -> list[float]:
    nums = re.findall(r"[-+]?\d*\.?\d+", s)
    return [float(n) for n in nums]
