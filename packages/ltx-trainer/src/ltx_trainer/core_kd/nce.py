"""Nonverbal Conflict Exposure (Sec. 4.4)."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ltx_trainer.core_kd.csa import cross_entropy

CONFLICT_SUBSETS: tuple[frozenset[str], ...] = (
    frozenset({"audio"}),
    frozenset({"video"}),
    frozenset({"audio", "video"}),
)


def donor_indices(labels: np.ndarray, target_idx: int) -> list[int]:
    y_i = int(labels[target_idx])
    return [j for j in range(labels.size) if int(labels[j]) != y_i]


def build_conflict_view(
    features: dict[str, np.ndarray],
    donor_features: dict[str, np.ndarray],
    replace: set[str],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for m, vec in features.items():
        out[m] = donor_features[m].copy() if m in replace else vec.copy()
    return out


def nce_loss(
    labels: np.ndarray,
    conflict_logits_fn,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[float, int]:
    """Mean CE over valid (i, donor, conflict subset) samples."""
    rng = rng or np.random.default_rng(0)
    labels = np.asarray(labels, dtype=np.int64)
    losses: list[float] = []
    for i in range(labels.size):
        donors = donor_indices(labels, i)
        if not donors:
            continue
        j = int(rng.choice(donors))
        b = CONFLICT_SUBSETS[int(rng.integers(0, len(CONFLICT_SUBSETS)))]
        logits = conflict_logits_fn(i, j, set(b))
        losses.append(cross_entropy(logits, int(labels[i])))
    if not losses:
        return 0.0, 0
    return float(np.mean(losses)), len(losses)


def rejection_rate(
    target_mu: np.ndarray,
    donor_mu: np.ndarray,
    conflict_mu: np.ndarray,
) -> bool:
    """True when conflict state is closer to target than donor (App. G.3)."""
    d_tar = float(np.linalg.norm(conflict_mu - target_mu))
    d_don = float(np.linalg.norm(conflict_mu - donor_mu))
    return d_tar < d_don


def mean_rejection_rate(pairs: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> float:
    flags = [rejection_rate(t, d, c) for t, d, c in pairs]
    return float(np.mean(flags)) if flags else 0.0
