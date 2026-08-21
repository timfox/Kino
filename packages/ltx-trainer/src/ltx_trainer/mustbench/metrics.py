"""MUSTBENCH evaluation metrics (Sec. 3.3)."""

from __future__ import annotations

from typing import Any

import numpy as np


def hit_at_t(
    pred_times: list[float],
    gt_times: list[float],
    *,
    tolerance_s: float = 3.0,
) -> float:
    """TSG Hit@T — fraction within tolerance."""
    if not pred_times or not gt_times:
        return 0.0
    n = min(len(pred_times), len(gt_times))
    hits = sum(1 for p, g in zip(pred_times[:n], gt_times[:n], strict=False) if abs(p - g) <= tolerance_s)
    return hits / n


def temporal_iou_f1(
    pred_intervals: list[tuple[float, float]],
    gt_intervals: list[tuple[float, float]],
) -> dict[str, float]:
    """MTR Temporal IoU and F1 over union of intervals."""

    def _union_length(intervals: list[tuple[float, float]]) -> float:
        if not intervals:
            return 0.0
        merged: list[tuple[float, float]] = []
        for s, e in sorted(intervals):
            if not merged or s > merged[-1][1]:
                merged.append((s, e))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        return sum(e - s for s, e in merged)

    def _intersection_length(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
        total = 0.0
        for s1, e1 in a:
            for s2, e2 in b:
                lo = max(s1, s2)
                hi = min(e1, e2)
                if hi > lo:
                    total += hi - lo
        return total

    pred_u = _union_length(pred_intervals)
    gt_u = _union_length(gt_intervals)
    inter = _intersection_length(pred_intervals, gt_intervals)
    union = pred_u + gt_u - inter
    iou = inter / union if union > 0 else 0.0
    f1 = 2 * inter / (pred_u + gt_u) if (pred_u + gt_u) > 0 else 0.0
    return {"iou": float(iou), "f1": float(f1)}


def mc_accuracy(preds: list[str], gts: list[str]) -> float:
    """LTR / GTO accuracy."""
    if not preds:
        return 0.0
    correct = sum(1 for p, g in zip(preds, gts, strict=False) if p.strip().upper() == g.strip().upper())
    return correct / len(preds)


def meteor_proxy(hypothesis: str, reference: str) -> float:
    """Lexical overlap proxy for METEOR (stub — word F1)."""
    hyp = set(hypothesis.lower().split())
    ref = set(reference.lower().split())
    if not hyp or not ref:
        return 0.0
    prec = len(hyp & ref) / len(hyp)
    rec = len(hyp & ref) / len(ref)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def clap_score_proxy(audio_emb: np.ndarray, text_emb: np.ndarray) -> float:
    """Cosine similarity proxy for CLAP Score."""
    a = audio_emb / (np.linalg.norm(audio_emb) + 1e-8)
    t = text_emb / (np.linalg.norm(text_emb) + 1e-8)
    return float(np.dot(a, t))


def metrics_smoke(*, seed: int = 42) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    tsg = hit_at_t([45.0, 120.0], [47.0, 118.0], tolerance_s=3.0)
    mtr = temporal_iou_f1([(70.0, 110.0)], [(72.0, 108.0)])
    acc = mc_accuracy(["A", "C"], ["A", "B"])
    met = meteor_proxy("drums surge forward", "the drums surge with punchier attack")
    clap = clap_score_proxy(rng.standard_normal(32), rng.standard_normal(32))
    return {"tsg_hit3": tsg, "mtr": mtr, "mc_acc": acc, "meteor_proxy": met, "clap_proxy": clap}
