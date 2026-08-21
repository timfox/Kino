"""Noise-aware champion validation (Appendix A.6, Eq. 1)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class NoiseGate:
    sigma: float | None = None
    pairs: list[tuple[float, float]] = field(default_factory=list)
    locked: bool = False
    default_sigma: float = 0.01
    band_multiplier: float = 2.0

    def promote(
        self,
        delta: float,
        *,
        confirm_fn: Callable[[], bool] | None = None,
    ) -> bool:
        """delta > 0 means improvement when metric is maximized; negate externally for minimization."""
        if self.sigma is None:
            band = self.default_sigma * self.band_multiplier
        else:
            band = self.sigma * self.band_multiplier

        if delta > band:
            return True
        if 0 < delta <= band:
            if confirm_fn is None:
                return False
            return confirm_fn()
        return False

    def record_duplicate_seed_pair(self, m1: float, m2: float) -> None:
        if self.locked:
            return
        self.pairs.append((m1, m2))
        if len(self.pairs) >= 3:
            n = len(self.pairs)
            self.sigma = (sum((a - b) ** 2 for a, b in self.pairs) / (2 * n)) ** 0.5
        if len(self.pairs) >= 5:
            self.locked = True
