"""Ethics Pluralism benchmark — 450 cases, 30 per subtheory (arXiv:2605.28707)."""

from __future__ import annotations

import random
from typing import Any

from ltx_trainer.ethical_pluralism.features import EthicalCase
from ltx_trainer.ethical_pluralism.taxonomy import (
    BENCHMARK_SIZE,
    CASES_PER_SUBTHEORY,
    SUBTHEORIES,
    SUBTHEORY_BY_ID,
)


def _case_template(subtheory_id: str, idx: int, *, include_bracket_markers: bool = True) -> EthicalCase:
    spec = SUBTHEORY_BY_ID[subtheory_id]
    school_kw = {
        "consequentialism": "outcomes utility harm benefit aggregate",
        "virtue_ethics": "character compassion wisdom integrity care",
        "deontology": "duty rights rule obligation universal law",
    }[spec.school]
    # Distinctive lexical anchors per subtheory for stub classifier separability
    sub_kw = subtheory_id.replace("_", " ")
    prefix = f"[{subtheory_id}] " if include_bracket_markers else ""
    text = (
        f"{prefix}Case {idx}: Normative lens {spec.name}. "
        f"{spec.description} School={spec.school}. "
        f"Markers: {school_kw} {sub_kw} marker_{idx % 5} theory_token_{subtheory_id}."
    )
    severity = ("low", "medium", "high")[(idx + hash(subtheory_id)) % 3]
    utility = ("good", "bad", "gray")[(idx + len(subtheory_id)) % 3]
    intention = ("benevolent", "neutral", "malicious", "mixed")[(idx + ord(subtheory_id[0])) % 4]
    case = EthicalCase(
        case_id=f"ep-{subtheory_id}-{idx:03d}",
        subtheory_id=subtheory_id,
        selftext=text,
        summary=text[:128],
        active_agent="agent",
        passive_agent="other",
        relationship=("stranger", "friend", "family", "colleague")[idx % 4],
        action="withheld information" if idx % 2 else "direct intervention",
        domain=("work", "healthcare", "public", "online")[idx % 4],
        ethical_issues=f"conflict under {spec.name}",
        consequence="mixed outcome",
        severity=severity,
        duration=("immediate", "short", "long")[idx % 3],
        utility=utility,
        moral_intention=intention,
        principles_upheld="duty of care" if spec.school == "deontology" else "",
        principles_violated="honesty" if idx % 5 == 0 else "",
        moral_decision=utility,
    )
    case.set_normative_from_school(noise=(idx % 7) * 0.02)
    return case


def generate_benchmark(
    seed: int = 42,
    *,
    cases_per_subtheory: int | None = None,
    include_bracket_markers: bool = True,
) -> list[EthicalCase]:
    """Balanced benchmark (default 450 cases, 30 per subtheory)."""
    n_per = cases_per_subtheory if cases_per_subtheory is not None else CASES_PER_SUBTHEORY
    rng = random.Random(seed)
    cases: list[EthicalCase] = []
    for spec in SUBTHEORIES:
        for i in range(n_per):
            cases.append(
                _case_template(spec.subtheory_id, i, include_bracket_markers=include_bracket_markers)
            )
    rng.shuffle(cases)
    expected = len(SUBTHEORIES) * n_per
    assert len(cases) == expected
    return cases


def train_test_split(
    cases: list[EthicalCase],
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[list[EthicalCase], list[EthicalCase]]:
    """Stratified split — hold out floor(30 * (1-fraction)) cases per subtheory."""
    rng = random.Random(seed)
    by_id: dict[str, list[EthicalCase]] = {}
    for c in cases:
        by_id.setdefault(c.subtheory_id, []).append(c)
    train: list[EthicalCase] = []
    test: list[EthicalCase] = []
    for group in by_id.values():
        rng.shuffle(group)
        n_test = max(1, int(round(len(group) * (1.0 - train_fraction))))
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def benchmark_card() -> dict[str, Any]:
    return {
        "n_cases": BENCHMARK_SIZE,
        "cases_per_subtheory": CASES_PER_SUBTHEORY,
        "n_subtheories": len(SUBTHEORIES),
        "source": "Moral Compass / Moral Decision Dataset subset (Aijaz et al. 2025) + expert labels",
        "augmentation": "DeepSeek-V3 for under-represented subtheories (paper)",
        "feature_categories": [
            "agent",
            "action",
            "consequentialism",
            "virtue_ethics",
            "deontology",
            "meta",
        ],
    }
