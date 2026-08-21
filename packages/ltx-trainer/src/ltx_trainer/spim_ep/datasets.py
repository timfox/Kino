"""Wine classification data for experimental SPIM-EP demo (Sec. IV)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def load_wine(*, seed: int = 0, train_frac: float = 0.8) -> dict[str, Any]:
    """UCI Wine dataset; normalized inputs in [-1, 1], one-hot targets in {−1, 1}."""
    try:
        from sklearn.datasets import load_wine as _load
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        data = _load()
        x = StandardScaler().fit_transform(data.data).astype(np.float64)
        x = np.clip(x / (np.max(np.abs(x)) + 1e-8), -1.0, 1.0)
        y_idx = data.target.astype(int)
        n_classes = int(y_idx.max()) + 1
        x_train, x_test, y_train, y_test = train_test_split(
            x, y_idx, train_size=train_frac, random_state=seed, stratify=y_idx
        )
        return {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_idx_to_onehot(y_train, n_classes),
            "y_test": y_idx_to_onehot(y_test, n_classes),
            "y_train_idx": y_train,
            "y_test_idx": y_test,
            "n_features": x.shape[1],
            "n_classes": n_classes,
            "source": "sklearn",
        }
    except ImportError:
        return synthetic_wine(seed=seed, train_frac=train_frac)


def idx_to_onehot(y: NDArray[np.integer], n_classes: int) -> NDArray[np.floating]:
    return y_idx_to_onehot(y, n_classes)


def y_idx_to_onehot(y: NDArray[np.integer], n_classes: int) -> NDArray[np.floating]:
    out = np.full((y.shape[0], n_classes), -1.0, dtype=np.float64)
    for i, c in enumerate(y):
        out[i, c] = 1.0
        out[i, [j for j in range(n_classes) if j != c]] = -1.0
    return out


def synthetic_wine(*, seed: int = 0, train_frac: float = 0.8) -> dict[str, Any]:
    """Fallback when sklearn is unavailable."""
    rng = np.random.default_rng(seed)
    n = 178
    n_features = 13
    n_classes = 3
    x = rng.normal(size=(n, n_features))
    y_idx = rng.integers(0, n_classes, size=n)
    x = np.clip(x / 3.0, -1.0, 1.0)
    n_train = int(n * train_frac)
    return {
        "x_train": x[:n_train],
        "x_test": x[n_train:],
        "y_train": y_idx_to_onehot(y_idx[:n_train], n_classes),
        "y_test": y_idx_to_onehot(y_idx[n_train:], n_classes),
        "y_train_idx": y_idx[:n_train],
        "y_test_idx": y_idx[n_train:],
        "n_features": n_features,
        "n_classes": n_classes,
        "source": "synthetic",
    }
