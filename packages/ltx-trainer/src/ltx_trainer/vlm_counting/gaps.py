"""Vision gap vs language gap diagnostics — § 2.3, Fig. 3."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CountingDiagnostics:
    ng: int
    nh: float
    np_: int

    @property
    def vision_gap(self) -> float:
        return abs(self.nh - self.ng)

    @property
    def language_gap(self) -> float:
        return abs(self.np_ - self.nh)


def regime_for_n(n: int, *, id_max: int = 49, ve_max: int = 99) -> str:
    if n <= id_max:
        return "ID"
    if n <= ve_max:
        return "VE"
    return "FE"


def synthetic_predicted_count(n: int, *, visual_train_max: int = 49, text_max: int = 99) -> int:
    """Toy decoder: perfect ID, collapse + attractors in VE/FE — Fig. 2, 5."""
    if n <= visual_train_max:
        return n
    if n == text_max:
        return text_max
    attractors = {49, 56, 58, 59, 90, 94, 95, 96, 97, 98, 99, 9}
    return 49 if n < 60 else (99 if n > 90 else 9)


def comparative_match_accuracy(n: int, *, visual_train_max: int = 49) -> float:
    """Vision-to-text comparison — Fig. 4."""
    if n <= visual_train_max:
        return 1.0
    if n <= 99:
        return 0.92
    return 0.55
