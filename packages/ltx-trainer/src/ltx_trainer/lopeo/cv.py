"""LOTO, LOPEO, and LOEO cross-validation partitioning."""

from __future__ import annotations

import random
from collections.abc import Hashable, Iterable, Sequence
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


def unordered_pair(a: T, b: T) -> tuple[T, T]:
    """Canonical unordered stimulus pair for LOPEO."""
    return (a, b) if a <= b else (b, a)


def trial_stimulus_pair(
    attended: T,
    unattended: T | Sequence[T],
) -> tuple[T, T] | None:
    """Two-speaker trials yield one pair; 3-speaker uses attended-only LOEO elsewhere."""
    if isinstance(unattended, Sequence) and not isinstance(unattended, (str, bytes)):
        if len(unattended) != 1:
            return None
        return unordered_pair(attended, unattended[0])
    return unordered_pair(attended, unattended)


def collect_stimulus_pairs(
    trials: Sequence[dict[str, object]],
    *,
    attended_key: str = "attended",
    unattended_key: str = "unattended",
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for trial in trials:
        pair = trial_stimulus_pair(
            str(trial[attended_key]),
            trial[unattended_key],  # type: ignore[arg-type]
        )
        if pair is not None:
            pairs.add((str(pair[0]), str(pair[1])))
    return pairs


def partition_trials_loto(
    trials: Sequence[dict[str, object]],
    *,
    trial_id_key: str = "trial_id",
) -> list[tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]]:
    """Leave-one-trial-out: each fold holds one trial out for test."""
    folds: list[tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]] = []
    n = len(trials)
    for i in range(n):
        test = [trials[i]]
        train_val = [t for j, t in enumerate(trials) if j != i]
        if len(train_val) >= 2:
            split = len(train_val) // 5 or 1
            val = train_val[:split]
            train = train_val[split:]
        else:
            val = []
            train = train_val
        folds.append((train, val, test))
    return folds


def partition_trials_lopeo(
    trials: Sequence[dict[str, object]],
    k: int,
    *,
    attended_key: str = "attended",
    unattended_key: str = "unattended",
    rng: random.Random | None = None,
) -> list[tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]]:
    """Algorithm 1 — partition by unordered attended–unattended stimulus pairs."""
    rng = rng or random.Random(0)
    pair_to_trials: dict[tuple[str, str], list[dict[str, object]]] = {}
    for trial in trials:
        pair = trial_stimulus_pair(str(trial[attended_key]), trial[unattended_key])  # type: ignore[arg-type]
        if pair is None:
            continue
        key = (str(pair[0]), str(pair[1]))
        pair_to_trials.setdefault(key, []).append(trial)

    pairs = list(pair_to_trials.keys())
    if not pairs:
        return []
    if len(pairs) == 1:
        only = pairs[0]
        return [([], [], list(pair_to_trials[only]))]

    rng.shuffle(pairs)
    k = max(2, min(k, len(pairs)))
    fold_pairs: list[list[tuple[str, str]]] = [[] for _ in range(k)]
    for idx, pair in enumerate(pairs):
        fold_pairs[idx % k].append(pair)

    partitions: list[tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]] = []
    for t_idx in range(k):
        test_pairs = set(fold_pairs[t_idx])
        for v_idx in range(k):
            if v_idx == t_idx:
                continue
            val_pairs = set(fold_pairs[v_idx])
            train_pairs = {p for i, fp in enumerate(fold_pairs) if i not in (t_idx, v_idx) for p in fp}
            train: list[dict[str, object]] = []
            val: list[dict[str, object]] = []
            test: list[dict[str, object]] = []
            for pair, group in pair_to_trials.items():
                if pair in test_pairs:
                    test.extend(group)
                elif pair in val_pairs:
                    val.extend(group)
                elif pair in train_pairs:
                    train.extend(group)
            partitions.append((train, val, test))
    return partitions


def partition_trials_loeo(
    trials: Sequence[dict[str, object]],
    k: int,
    *,
    attended_key: str = "attended",
    rng: random.Random | None = None,
) -> list[tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]]:
    """Leave-one-envelope-out (NJU cEEGrid): withhold attended stimulus identity only."""
    rng = rng or random.Random(0)
    att_to_trials: dict[str, list[dict[str, object]]] = {}
    for trial in trials:
        a = str(trial[attended_key])
        att_to_trials.setdefault(a, []).append(trial)

    stimuli = list(att_to_trials.keys())
    rng.shuffle(stimuli)
    k = max(1, min(k, len(stimuli) or 1))
    fold_atts: list[list[str]] = [[] for _ in range(k)]
    for idx, att in enumerate(stimuli):
        fold_atts[idx % k].append(att)

    partitions: list[tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]] = []
    for t_idx in range(k):
        test_atts = set(fold_atts[t_idx])
        for v_idx in range(k):
            if v_idx == t_idx:
                continue
            val_atts = set(fold_atts[v_idx])
            train_atts = {a for i, fa in enumerate(fold_atts) if i not in (t_idx, v_idx) for a in fa}
            train, val, test = [], [], []
            for att, group in att_to_trials.items():
                if att in test_atts:
                    test.extend(group)
                elif att in val_atts:
                    val.extend(group)
                elif att in train_atts:
                    train.extend(group)
            partitions.append((train, val, test))
    return partitions


def pair_leakage_count(
    train_trials: Iterable[dict[str, object]],
    test_trials: Iterable[dict[str, object]],
    *,
    attended_key: str = "attended",
    unattended_key: str = "unattended",
) -> int:
    """Count test stimulus pairs also present in train (LOTO may leak; LOPEO should be 0)."""
    def pairs(ts: Iterable[dict[str, object]]) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        for t in ts:
            p = trial_stimulus_pair(str(t[attended_key]), t[unattended_key])  # type: ignore[arg-type]
            if p is not None:
                out.add((str(p[0]), str(p[1])))
        return out

    train_p = pairs(train_trials)
    test_p = pairs(test_trials)
    return len(train_p & test_p)
