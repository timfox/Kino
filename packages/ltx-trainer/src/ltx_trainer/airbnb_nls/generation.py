"""Contrastive synthetic query generation (Algorithm 1, Sec. 3)."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

from ltx_trainer.airbnb_nls.config import PLATFORM_TERMS_BLOCKLIST, AirbnbNLSConfig


@dataclass
class ListingFeatures:
    """Structured listing features for prompt grounding."""

    title: str
    description: str
    amenities: list[str]
    property_type: str
    location_hint: str
    bedrooms: int | None = None
    rating: float | None = None


@dataclass
class ContrastiveTriplet:
    """Labeled (q, l+, l-) with topicality by construction."""

    query: str
    positive: ListingFeatures
    negative: ListingFeatures
    seed_template: str
    prompt_variant: str


def contains_platform_term(query: str) -> bool:
    q = query.lower()
    return any(term in q for term in PLATFORM_TERMS_BLOCKLIST)


def apply_blocklist(query: str) -> str:
    """Strip platform-internal terms (Appendix D)."""
    out = query
    for term in PLATFORM_TERMS_BLOCKLIST:
        out = re.sub(re.escape(term), "", out, flags=re.IGNORECASE)
    return " ".join(out.split())


def contrastive_generate_step(
    *,
    seed_queries: list[str],
    positive: ListingFeatures,
    negative: ListingFeatures,
    prompt_variant: str = "seed_controlled",
    rng: random.Random | None = None,
) -> ContrastiveTriplet:
    """Algorithm 1 — one synthetic triplet (LLM step stubbed as template transform)."""
    rng = rng or random.Random(0)
    template = rng.choice(seed_queries) if seed_queries else "pet friendly near beach"
    # Demo: terse seed-guided rewrite (production uses LLM)
    if prompt_variant == "seed_controlled":
        words = template.lower().split()[:6]
        amenity = positive.amenities[0] if positive.amenities else "wifi"
        query = " ".join(words[:3] + [amenity, "near", positive.location_hint.split()[0].lower()])
    elif prompt_variant == "seed_freeform":
        query = f"{positive.property_type} {positive.amenities[0] if positive.amenities else 'stay'}"
    else:
        query = f"{positive.location_hint} {rng.choice(['retreat', 'escape', 'getaway'])}"
    query = apply_blocklist(query)
    words = query.split()
    query = " ".join(words[:8]) if len(words) > 8 else query
    return ContrastiveTriplet(
        query=query,
        positive=positive,
        negative=negative,
        seed_template=template,
        prompt_variant=prompt_variant,
    )


def demo_listing_pair() -> tuple[ListingFeatures, ListingFeatures]:
    """Figure 2 style listing pair."""
    pos = ListingFeatures(
        title="Charming studio near beach",
        description="Modern kitchenette, outdoor seating, Wi-Fi near boardwalk.",
        amenities=["wifi", "outdoor seating", "kitchenette"],
        property_type="studio",
        location_hint="near beach boardwalk",
    )
    neg = ListingFeatures(
        title="Downtown apartment",
        description="Urban loft without outdoor space.",
        amenities=["wifi", "parking"],
        property_type="apartment",
        location_hint="downtown",
    )
    return pos, neg


def generation_modes_catalog(cfg: AirbnbNLSConfig | None = None) -> list[dict[str, str]]:
    cfg = cfg or AirbnbNLSConfig()
    return [
        {
            "name": "seed_controlled",
            "description": "Template-based; preserves seed query structure (Sec. 3.4).",
        },
        {
            "name": "seed_freeform",
            "description": "Few-shot style examples from seeds; flexible attributes.",
        },
        {
            "name": "variety",
            "description": "Relaxed constraints, higher diversity (20% production mix).",
        },
    ]
