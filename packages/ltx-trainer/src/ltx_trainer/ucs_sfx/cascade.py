"""Four-stage tag-to-UCS cascade (Sec. 2.1.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def normalize_tag(tag: str) -> str:
    return tag.lower().replace("_", " ").strip()


@dataclass(frozen=True)
class UcsMatch:
    category: str
    subcategory: str | None = None
    stage: str = "unclassified"
    specificity: int = 0  # 2=subcat, 1=category, 0=mapping/synonym default


def stage1_predefined(tag: str, mapping: dict[str, tuple[str, str | None]]) -> UcsMatch | None:
    key = normalize_tag(tag)
    if key in mapping:
        cat, sub = mapping[key]
        return UcsMatch(category=cat, subcategory=sub, stage="predefined", specificity=2 if sub else 1)
    return None


def stage2_subcategory(tag: str, subcategories: dict[str, str]) -> UcsMatch | None:
    key = normalize_tag(tag)
    if key in subcategories:
        return UcsMatch(category=subcategories[key], subcategory=key.upper(), stage="subcategory", specificity=2)
    return None


def stage3_category(tag: str, categories: set[str]) -> UcsMatch | None:
    key = normalize_tag(tag).upper()
    if key in categories:
        return UcsMatch(category=key, subcategory=None, stage="category", specificity=1)
    return None


def stage4_synonym(tag: str, synonyms: dict[str, tuple[str, str]]) -> UcsMatch | None:
    key = normalize_tag(tag)
    if key in synonyms:
        cat, sub = synonyms[key]
        return UcsMatch(category=cat, subcategory=sub, stage="synonym", specificity=2)
    return None


def classify_tag(
    tag: str,
    *,
    mapping: dict[str, tuple[str, str | None]] | None = None,
    subcategories: dict[str, str] | None = None,
    categories: set[str] | None = None,
    synonyms: dict[str, tuple[str, str]] | None = None,
) -> UcsMatch:
    mapping = mapping or {}
    subcategories = subcategories or {}
    categories = categories or set()
    synonyms = synonyms or {}
    for fn, args in (
        (stage1_predefined, (tag, mapping)),
        (stage2_subcategory, (tag, subcategories)),
        (stage3_category, (tag, categories)),
        (stage4_synonym, (tag, synonyms)),
    ):
        hit = fn(*args)
        if hit is not None:
            return hit
    return UcsMatch(category="", subcategory=None, stage="unclassified", specificity=0)


def classify_tags(tags: list[str], **kwargs: Any) -> list[UcsMatch]:
    return [classify_tag(t, **kwargs) for t in tags]
