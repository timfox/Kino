"""Flattened 3DGS splat parameters for evolution-tree refinement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SplatVector:
    """Single splat as a flat vector (position, rotation, scale, opacity, SH DC)."""

    values: np.ndarray

    @classmethod
    def zeros(cls, dim: int = 59) -> SplatVector:
        v = np.zeros(dim, dtype=np.float64)
        v[-1] = 0.5
        return cls(v)

    @property
    def opacity(self) -> float:
        return float(self.values[-1])

    def with_opacity(self, opacity: float) -> SplatVector:
        out = self.values.copy()
        out[-1] = float(np.clip(opacity, 1e-4, 1.0 - 1e-4))
        return SplatVector(out)


def random_splats(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n splat vectors with small scales and mid opacity."""
    x = rng.normal(size=(n, dim)).astype(np.float64) * 0.05
    x[:, -1] = rng.uniform(0.2, 0.9, size=n)
    return x


def ghost_mask(opacities: np.ndarray, threshold: float = 0.005) -> np.ndarray:
    return opacities < threshold
