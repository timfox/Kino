"""Deterministic verification metrics (r, RMSE, bias, MAE)."""

from __future__ import annotations

import numpy as np


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-12:
        return 0.0
    return float((a * b).sum() / denom)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def bias(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def skill_row(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    return {
        "r": round(pearson_r(pred, ref), 4),
        "rmse": round(rmse(pred, ref), 4),
        "bias": round(bias(pred, ref), 4),
        "mae": round(mae(pred, ref), 4),
    }
