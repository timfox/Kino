"""Model merging stubs: Task Arithmetic, TIES, DARE."""

from __future__ import annotations

from enum import Enum

import numpy as np


class MergeMethod(str, Enum):
    TASK_ARITHMETIC = "task_arithmetic"
    TIES = "ties"
    DARE = "dare"


def task_arithmetic(deltas: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Linear sum of task vectors (Ilharco et al., 2023)."""
    if not deltas:
        raise ValueError("deltas must be non-empty")
    w = weights or [1.0 / len(deltas)] * len(deltas)
    out = np.zeros_like(deltas[0])
    for d, wi in zip(deltas, w):
        out = out + wi * d
    return out


def ties_merge(deltas: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Conflict-aware sparse sign agreement merge (Yadav et al., 2023)."""
    if len(deltas) == 1:
        return deltas[0].copy()
    stacked = np.stack(deltas, axis=0)
    signs = np.sign(stacked)
    sign_sum = signs.sum(axis=0)
    mask = np.abs(sign_sum) == len(deltas)
    avg = stacked.mean(axis=0)
    merged = np.where(mask, avg, 0.0)
    if weights:
        scale = sum(weights) / len(weights)
        merged = merged * scale
    return merged


def dare_merge(
    deltas: list[np.ndarray],
    drop_rate: float = 0.9,
    rescale: float = 1.0 / 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Random prune + rescale merge (Yu et al., 2024)."""
    rng = np.random.default_rng(seed)
    merged = np.zeros_like(deltas[0])
    for d in deltas:
        mask = rng.random(d.shape) > drop_rate
        pruned = np.where(mask, d, 0.0) * rescale
        merged = merged + pruned / len(deltas)
    return merged


def merge_models(
    method: MergeMethod,
    deltas: list[np.ndarray],
    *,
    seed: int = 0,
) -> np.ndarray:
    if method == MergeMethod.TASK_ARITHMETIC:
        return task_arithmetic(deltas)
    if method == MergeMethod.TIES:
        return ties_merge(deltas)
    return dare_merge(deltas, seed=seed)
