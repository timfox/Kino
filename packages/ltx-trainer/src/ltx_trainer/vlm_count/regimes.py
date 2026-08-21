"""Counting evaluation regimes: ID, visual extrapolation (VE), full extrapolation (FE)."""

from __future__ import annotations

from typing import Literal

Regime = Literal["ID", "VE", "FE"]


def regime_for_n(n: int, *, visual_train_max: int = 49, full_extrap_max: int = 120) -> Regime:
    if n <= visual_train_max:
        return "ID"
    if n <= full_extrap_max:
        if n <= 99:
            return "VE"
        return "FE"
    return "FE"


def regime_label(regime: Regime) -> str:
    labels = {
        "ID": "in_distribution",
        "VE": "visual_extrapolation",
        "FE": "full_extrapolation",
    }
    return labels[regime]
