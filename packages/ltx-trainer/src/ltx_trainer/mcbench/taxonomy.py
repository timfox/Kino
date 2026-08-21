"""Four coarse-grained safety taxonomies (Sec. 2.1, Table 2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyCategory:
    name: str
    samples: int
    percentage: float
    fine_grained: tuple[str, ...]


CATEGORIES: tuple[SafetyCategory, ...] = (
    SafetyCategory(
        "physical_harm",
        300,
        25.1,
        ("violence_weaponized_threats", "environment_hazards", "public_safety_medical"),
    ),
    SafetyCategory(
        "social_harm",
        300,
        25.1,
        ("identity_exclusion", "information_violations", "exploitation_interpersonal"),
    ),
    SafetyCategory(
        "illegal_harm",
        296,
        24.7,
        ("crimes_against_people", "crimes_against_systems_property"),
    ),
    SafetyCategory(
        "property_damage",
        300,
        25.1,
        ("energy_utility_hazards", "environmental_structural", "industrial_vehicular"),
    ),
)


def total_samples() -> int:
    return sum(c.samples for c in CATEGORIES)


def category_by_name(name: str) -> SafetyCategory:
    return next(c for c in CATEGORIES if c.name == name)
