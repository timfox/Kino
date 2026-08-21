"""RATE-style asynchronous feature tracker stub (Sec. III, [8])."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class FeatureUpdate:
    """hj = (id_j, t_j, (u_j, v_j)) from RATE ROS messages."""

    feature_id: int
    time_s: float
    u: float
    v: float


@dataclass
class RateTrackerStub:
    """
    Lightweight stand-in for RATE: Shi-Tomasi on SAEB + async HASTE-like tracks.

    Generates synthetic corner tracks on a moving intensity pattern for smoke tests.
    """

    width: int = 640
    height: int = 480
    n_features: int = 80
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._ids = list(range(self.n_features))
        self._uv = rng.uniform([20, 20], [self.width - 20, self.height - 20], size=(self.n_features, 2))
        self._vel = rng.normal(scale=0.3, size=(self.n_features, 2))

    def stream_updates(
        self,
        times: np.ndarray,
        *,
        global_shift: np.ndarray | None = None,
    ) -> Iterator[FeatureUpdate]:
        """Emit one update per feature per timestep (async order shuffled)."""
        rng = np.random.default_rng(self.seed + 1)
        for t in times:
            shift = global_shift if global_shift is not None else np.zeros(2)
            order = rng.permutation(self.n_features)
            for i in order:
                self._uv[i] += self._vel[i] * 0.05 + shift
                self._uv[i, 0] = np.clip(self._uv[i, 0], 5, self.width - 5)
                self._uv[i, 1] = np.clip(self._uv[i, 1], 5, self.height - 5)
                yield FeatureUpdate(self._ids[i], float(t), float(self._uv[i, 0]), float(self._uv[i, 1]))
