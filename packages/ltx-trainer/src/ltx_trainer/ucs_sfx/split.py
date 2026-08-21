"""UCS-aware stratified split stub (Sec. 2.2)."""

from __future__ import annotations

import numpy as np


def composite_key(category: str, subcategory: str | None) -> str:
    return f"{category}||{subcategory or ''}"


def stratified_split_indices(
    keys: list[str],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    min_stratify: int = 5,
) -> dict[str, list[int]]:
    """Two-pass 70/15/15 split; stratify within each composite key when possible."""
    rng = np.random.default_rng(seed)
    by_key: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        by_key.setdefault(key, []).append(i)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for idxs in by_key.values():
        if len(idxs) < min_stratify:
            train_idx.extend(idxs)
            continue
        perm = idxs.copy()
        rng.shuffle(perm)
        n = len(perm)
        n_test = int(round(n * (1.0 - train_ratio - val_ratio)))
        n_val = int(round(n * val_ratio))
        n_test = max(0, min(n_test, n))
        n_val = max(0, min(n_val, n - n_test))
        test_idx.extend(perm[:n_test])
        val_idx.extend(perm[n_test : n_test + n_val])
        train_idx.extend(perm[n_test + n_val :])

    return {
        "train": sorted(train_idx),
        "val": sorted(val_idx),
        "test": sorted(test_idx),
    }


def split_distribution_correlation(
    keys: list[str],
    train_idx: list[int],
    test_idx: list[int],
) -> float:
    """Pearson r between train and test category marginals."""
    train_cats = [keys[i].split("||")[0] for i in train_idx]
    test_cats = [keys[i].split("||")[0] for i in test_idx]
    all_cats = sorted(set(train_cats) | set(test_cats))
    if len(all_cats) < 2:
        return 1.0
    tr = np.array([train_cats.count(c) for c in all_cats], dtype=np.float64)
    te = np.array([test_cats.count(c) for c in all_cats], dtype=np.float64)
    tr /= tr.sum() + 1e-8
    te /= te.sum() + 1e-8
    if np.std(tr) < 1e-8 or np.std(te) < 1e-8:
        return 1.0
    return float(np.corrcoef(tr, te)[0, 1])
