"""Topicality label generation: contrastive + Virtual Judge (Sec. 4)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.airbnb_nls.generation import ContrastiveTriplet, ListingFeatures


@dataclass
class TopicalityLabel:
    """Query-listing topicality judgment."""

    query: str
    listing_id: str
    is_more_relevant: bool
    source: str  # "contrastive" | "virtual_judge"
    confidence: float = 1.0


def contrastive_label(triplet: ContrastiveTriplet) -> tuple[TopicalityLabel, TopicalityLabel]:
    """Labels by construction: l+ preferred over l- for q (Sec. 4.1)."""
    q = triplet.query
    pos = TopicalityLabel(q, "l_plus", True, "contrastive", 1.0)
    neg = TopicalityLabel(q, "l_minus", False, "contrastive", 1.0)
    return pos, neg


def virtual_judge_score(
    query: str,
    listing: ListingFeatures,
    *,
    reference: ListingFeatures | None = None,
) -> float:
    """Heuristic VJ: attribute overlap with query tokens (demo; production uses LLM)."""
    q_tokens = set(query.lower().split())
    attrs = " ".join(listing.amenities + [listing.property_type, listing.location_hint]).lower()
    attr_tokens = set(attrs.split())
    overlap = len(q_tokens & attr_tokens) / max(1, len(q_tokens))
    if reference is not None:
        ref_attrs = " ".join(reference.amenities + [reference.property_type]).lower()
        ref_overlap = len(q_tokens & set(ref_attrs.split())) / max(1, len(q_tokens))
        return overlap - ref_overlap
    return overlap


def virtual_judge_pairwise(
    query: str,
    listing_a: ListingFeatures,
    listing_b: ListingFeatures,
) -> bool:
    """True if listing_a is more topical than listing_b."""
    return virtual_judge_score(query, listing_a) >= virtual_judge_score(query, listing_b)


def llm_self_preference_accuracy_table() -> dict[str, float]:
    """Table 4 — GPT evaluator on GPT-generated pairs."""
    return {"gpt_4": 0.994, "gpt_4o": 0.988, "gpt_5": 0.990}
