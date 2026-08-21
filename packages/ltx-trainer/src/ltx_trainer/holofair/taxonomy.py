"""FairFace-aligned demographic taxonomy (Sec. 3.1.1, Appendix C.3.1)."""

from __future__ import annotations

from ltx_trainer.holofair.config import HoloFairConfig


def age_label(years: int) -> str:
    """Map continuous age to {young, middle, elderly}."""
    if years <= 29:
        return "young"
    if years <= 59:
        return "middle"
    return "elderly"


def empty_counts(cfg: HoloFairConfig | None = None) -> dict[str, dict[str, int]]:
    """Zero counts for all attributes and categories."""
    cfg = cfg or HoloFairConfig()
    return {a: {c: 0 for c in cfg.categories_for(a)} for a in cfg.attributes}


def counts_from_labels(
    labels: dict[str, str],
    *,
    cfg: HoloFairConfig | None = None,
) -> dict[str, dict[str, int]]:
    """Increment counts from one image's predicted labels."""
    cfg = cfg or HoloFairConfig()
    counts = empty_counts(cfg)
    for attr in cfg.attributes:
        cat = labels.get(attr)
        if cat in counts[attr]:
            counts[attr][cat] += 1
    return counts
