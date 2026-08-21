"""Objective metrics stub (Sec. 3.3)."""

from __future__ import annotations


def delta_eecs(baseline: float, proposed: float) -> float:
    return proposed - baseline


def meets_identity_floor(secs_w: float, *, floor: float = 0.88) -> bool:
    return secs_w >= floor
