"""Metrics for SPIM-EP classification."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def accuracy(y_true: NDArray[np.integer], y_pred: NDArray[np.integer]) -> float:
    return float(np.mean(y_true == y_pred))


def mse_cost(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> float:
    return float(np.mean((y_true - y_pred) ** 2))
