"""Normative ethics taxonomy — Table 1 (arXiv:2605.28707)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

NormativeSchool = Literal["consequentialism", "virtue_ethics", "deontology"]

SCHOOL_ALPHA: dict[NormativeSchool, str] = {
    "consequentialism": "alpha",
    "virtue_ethics": "beta",
    "deontology": "gamma",
}


@dataclass(frozen=True)
class SubtheorySpec:
    subtheory_id: str
    school: NormativeSchool
    name: str
    description: str


SUBTHEORIES: tuple[SubtheorySpec, ...] = (
    # Consequentialism (α)
    SubtheorySpec(
        "act_utilitarianism",
        "consequentialism",
        "Act Utilitarianism",
        "Evaluates action based on maximization of utility.",
    ),
    SubtheorySpec(
        "rule_utilitarianism",
        "consequentialism",
        "Rule Utilitarianism",
        "Judges actions according to rules that produce the greatest good.",
    ),
    SubtheorySpec(
        "preference_utilitarianism",
        "consequentialism",
        "Preference Utilitarianism",
        "Satisfies preferences of individuals.",
    ),
    SubtheorySpec(
        "negative_utilitarianism",
        "consequentialism",
        "Negative Utilitarianism",
        "Prioritizes minimal suffering over maximal happiness.",
    ),
    SubtheorySpec(
        "ethical_egoism",
        "consequentialism",
        "Ethical Egoism",
        "Advances self-interest of the decision-maker.",
    ),
    # Deontology (γ)
    SubtheorySpec(
        "kantian_deontology",
        "deontology",
        "Kantian Deontology",
        "Emphasizes universal moral duties.",
    ),
    SubtheorySpec(
        "ross_prima_facie",
        "deontology",
        "Ross's Prima Facie Duties",
        "Multiple competing duties balanced contextually.",
    ),
    SubtheorySpec(
        "divine_command",
        "deontology",
        "Divine Command Theory",
        "Adherence to commands from divine authority.",
    ),
    SubtheorySpec(
        "contractualism",
        "deontology",
        "Contractualism",
        "Actions justified by mutually acceptable principles.",
    ),
    SubtheorySpec(
        "rights_based_deontology",
        "deontology",
        "Rights-Based Deontology",
        "Prioritizes protection of individual rights.",
    ),
    # Virtue ethics (β)
    SubtheorySpec(
        "aristotelian_virtue",
        "virtue_ethics",
        "Aristotelian Virtue Ethics",
        "Virtuous character and practical wisdom.",
    ),
    SubtheorySpec(
        "stoic_virtue",
        "virtue_ethics",
        "Stoic Virtue Ethics",
        "Rational self-control and moral discipline.",
    ),
    SubtheorySpec(
        "confucian_virtue",
        "virtue_ethics",
        "Confucian Virtue Ethics",
        "Social harmony and respect within society.",
    ),
    SubtheorySpec(
        "thomistic_virtue",
        "virtue_ethics",
        "Thomistic Virtue Ethics",
        "Aristotelian virtues integrated with theological principles.",
    ),
    SubtheorySpec(
        "ethics_of_care",
        "virtue_ethics",
        "Ethics of Care",
        "Empathy, compassion, responsiveness to others' needs.",
    ),
)

SUBTHEORY_IDS: tuple[str, ...] = tuple(s.subtheory_id for s in SUBTHEORIES)
SUBTHEORY_BY_ID: dict[str, SubtheorySpec] = {s.subtheory_id: s for s in SUBTHEORIES}
CASES_PER_SUBTHEORY = 30
BENCHMARK_SIZE = len(SUBTHEORIES) * CASES_PER_SUBTHEORY  # 450
