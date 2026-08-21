"""Sixteen grapevine leaf traits (§2.1)."""

from __future__ import annotations

TRAIT_NAMES: tuple[str, ...] = (
    "N",
    "P",
    "K",
    "Ca",
    "Mg",
    "Mn",
    "Zn",
    "Fe",
    "Cu",
    "B",
    "EWT",
    "LMA",
    "Ant",
    "Car",
    "Nstruct",
    "Chl",
)

NUTRIENT_TRAITS: tuple[str, ...] = TRAIT_NAMES[:10]
STRUCTURAL_TRAITS: tuple[str, ...] = ("EWT", "LMA", "Ant", "Car", "Nstruct", "Chl")


def trait_index(name: str) -> int:
    return TRAIT_NAMES.index(name)
