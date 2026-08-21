"""Bridge Airbnb NL search listing features to Utility-Aware CLIP demand (Airbnb platform)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.airbnb_nls.generation import ContrastiveTriplet, ListingFeatures, contrastive_generate_step, demo_listing_pair
from ltx_trainer.uaclip.demand import airbnb_demand_score
from ltx_trainer.uaclip.visual_attrs import VisualAttributes

_OUTDOOR_AMENITIES = frozenset(
    {"outdoor seating", "pool", "garden", "patio", "balcony", "beach access", "hot tub"}
)


def listing_to_visual_attrs(listing: ListingFeatures) -> VisualAttributes:
    """Heuristic visual proxy from structured listing text (no image encoder)."""
    text = f"{listing.title} {listing.description} {listing.location_hint}".lower()
    outdoor = sum(1 for a in listing.amenities if a.lower() in _OUTDOOR_AMENITIES)
    beach = "beach" in text or "coast" in text
    studio = listing.property_type.lower() in {"studio", "cottage", "cabin"}
    urban = "downtown" in text or "urban" in text
    # Map text cues to Table 3 inverted-U peaks (uniqueness ~0.60, aesthetic ~0.48).
    if beach or (studio and outdoor):
        return VisualAttributes(uniqueness=0.58, aesthetic=0.50)
    if urban:
        return VisualAttributes(uniqueness=0.36, aesthetic=0.44)
    rating = float(listing.rating if listing.rating is not None else 4.2)
    aesthetic = float(min(0.52, max(0.40, (rating - 3.0) / 2.0 + 0.42)))
    raw_unique = 0.40 + 0.04 * len(listing.amenities) + 0.06 * outdoor
    uniqueness = float(min(0.58, max(0.34, raw_unique)))
    return VisualAttributes(uniqueness=uniqueness, aesthetic=aesthetic)


def score_listing_demand(listing: ListingFeatures) -> dict[str, Any]:
    attrs = listing_to_visual_attrs(listing)
    d = attrs.to_dict("airbnb")
    return {
        "listing_title": listing.title,
        "attributes": d,
        "demand_score": airbnb_demand_score(d),
    }


def score_contrastive_triplet(triplet: ContrastiveTriplet) -> dict[str, Any]:
    """Rank positive vs negative listing by U-CLIP Airbnb demand proxy."""
    pos = score_listing_demand(triplet.positive)
    neg = score_listing_demand(triplet.negative)
    return {
        "query": triplet.query,
        "prompt_variant": triplet.prompt_variant,
        "positive": pos,
        "negative": neg,
        "uclip_prefers_positive": pos["demand_score"] > neg["demand_score"],
        "demand_margin": pos["demand_score"] - neg["demand_score"],
    }


def airbnb_nls_uaclip_demo(*, seed: int = 0) -> dict[str, Any]:
    """One contrastive NLS triplet + Utility-Aware demand alignment check."""
    pos, neg = demo_listing_pair()
    triplet = contrastive_generate_step(
        seed_queries=["pet friendly near beach", "quiet wifi studio"],
        positive=pos,
        negative=neg,
        prompt_variant="seed_controlled",
    )
    scored = score_contrastive_triplet(triplet)
    return {
        "paper_uaclip": "2605.28733",
        "paper_airbnb_nls": "2605.21812",
        "bridge": "listing text → VisualAttributes → airbnb_demand_score",
        "triplet_scored": scored,
        "aligns_with_contrastive_label": scored["uclip_prefers_positive"],
    }
