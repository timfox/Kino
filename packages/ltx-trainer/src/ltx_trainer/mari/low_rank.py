"""Low-rank representation editor Φ(h) = h + γ s Δ(h) (Eq. 2–3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LowRankAdapter:
    """Rank-r adapter at a single injection site."""

    U: np.ndarray  # (d, r)
    V: np.ndarray  # (d, r)
    b: np.ndarray  # (r,)
    scale: float = 1.0

    @property
    def rank(self) -> int:
        return int(self.U.shape[1])

    def delta(self, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=np.float64).reshape(-1)
        z = self.V.T @ h + self.b
        return self.U @ z

    def intervene(
        self,
        h: np.ndarray,
        *,
        gamma: float = 1.0,
        alpha: float = 1.0,
    ) -> np.ndarray:
        return np.asarray(h, dtype=np.float64) + alpha * gamma * self.scale * self.delta(h)

    def output_subspace(self) -> np.ndarray:
        """Orthonormal basis Q for span(U)."""
        q, _ = np.linalg.qr(self.U, mode="reduced")
        return q


def init_adapter(d: int, r: int, rng: np.random.Generator, scale: float = 0.02) -> LowRankAdapter:
    U = rng.standard_normal((d, r)) * scale
    V = rng.standard_normal((d, r)) * scale
    b = rng.standard_normal(r) * scale
    return LowRankAdapter(U=U, V=V, b=b, scale=1.0)
