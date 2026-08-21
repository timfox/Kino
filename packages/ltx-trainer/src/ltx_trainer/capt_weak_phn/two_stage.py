"""Two-stage weak supervision and selection stubs (toy).

The paper studies:
- stage 1: train with utterance labels (1S-U)
- stage 2: select n utterances for word/phoneme labeling; fine-tune (2S FT) vs train from scratch (2S TR)
- selection variants: random, best-by-absolute-error, balanced vs unbalanced.

This module implements only the selection mechanics + a tiny "finetune gain" simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SelectionResult:
    selected_ids: list[int]
    strategy: str
    balanced: bool


def select_balanced_by_bins(
    ids: list[int],
    scores: np.ndarray,
    n: int,
    bins: int = 5,
    key: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Balanced selection by binning ground-truth utterance scores.

    - If key is None: select uniformly at random within each bin.
    - Else: select smallest key (e.g. absolute error) within each bin.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if n <= 0 or not ids:
        return []
    x = np.asarray(scores, dtype=np.float64)
    if x.shape[0] != len(ids):
        raise ValueError("scores must align with ids")
    k = np.asarray(key, dtype=np.float64) if key is not None else None

    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        # Degenerate: everything in one bin.
        order = np.argsort(k) if k is not None else rng.permutation(len(ids))
        return [ids[int(i)] for i in order[: min(n, len(ids))]]

    edges = np.linspace(lo, hi, bins + 1)
    per = int(np.ceil(n / bins))
    selected: list[int] = []
    for b in range(bins):
        in_bin = (x >= edges[b]) & (x <= edges[b + 1] if b == bins - 1 else x < edges[b + 1])
        idx = np.nonzero(in_bin)[0]
        if idx.size == 0:
            continue
        if k is None:
            pick = rng.choice(idx, size=min(per, idx.size), replace=False)
        else:
            pick = idx[np.argsort(k[idx])[: min(per, idx.size)]]
        selected.extend([ids[int(i)] for i in pick])
    # Trim if we overshot.
    if len(selected) > n:
        selected = selected[:n]
    return selected


def select_unbalanced(
    ids: list[int],
    n: int,
    key: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> list[int]:
    if rng is None:
        rng = np.random.default_rng(0)
    if n <= 0 or not ids:
        return []
    if key is None:
        perm = rng.permutation(len(ids))
        return [ids[int(i)] for i in perm[: min(n, len(ids))]]
    k = np.asarray(key, dtype=np.float64)
    if k.shape[0] != len(ids):
        raise ValueError("key must align with ids")
    order = np.argsort(k)
    return [ids[int(i)] for i in order[: min(n, len(ids))]]


def simulate_finetune_gain(
    base_pcc: float,
    n_labeled: int,
    method: str = "FT",
) -> float:
    """Toy monotone gain curve used by the smoke demo."""
    n = max(0, int(n_labeled))
    start = float(base_pcc)
    if method.upper() == "TR":
        # Training from scratch is worse at small budgets.
        gain = 0.20 * (1.0 - np.exp(-n / 900.0))
        penalty = 0.06 * np.exp(-n / 200.0)
        return float(min(0.95, start + gain - penalty))
    gain = 0.22 * (1.0 - np.exp(-n / 450.0))
    return float(min(0.95, start + gain))
