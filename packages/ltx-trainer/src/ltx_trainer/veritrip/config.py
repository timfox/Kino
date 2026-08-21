"""VeriTrip configuration (arXiv:2605.28683)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VeriTripConfig:
    paper_arxiv: str = "2605.28683"
    paper_title: str = "VeriTrip: A Verifiable Benchmark for Travel Planning Agents over Unstructured Web Corpora"

    # MRB corpus (Table 2)
    mrb_documents: int = 8210
    mrb_images: int = 4146
    mrb_cities: int = 15

    # VKB (Table 2)
    vkb_transportation: int = 57773
    vkb_restaurants: int = 4304
    vkb_accommodations: int = 3613
    vkb_attractions: int = 1056
    vkb_pois: int = 7400

    # Queries
    query_count: int = 228
    query_topics: tuple[str, ...] = (
        "transportation",
        "restaurants",
        "accommodations",
        "attractions",
        "confusing_attractions",
        "travel_guides",
        "pois",
        "user_comments",
    )

    # Retrieval (Appendix C)
    doc_search_top_k: int = 10
    snippet_max_tokens: int = 100
    fuzzy_match_threshold: float = 0.85
    budget_tolerance: float = 0.10
    inner_city_time_ratio_limit: float = 0.35

    # Geographic coherence (Sec. 3.4) — AM in units of 10 km
    am_scale_10km: float = 10.0

    # Snapshot window (Appendix B)
    snapshot_start: str = "2025-10-15"
    snapshot_end: str = "2025-10-30"


@dataclass
class VeriTripQuery:
    """Agent input Q = (qtxt, qimg)."""

    query_id: str
    text: str
    image_path: str | None = None
    start_city: str = ""
    target_city: str = ""
    start_date: str = ""
    end_date: str = ""
    people_number: int = 1
    budget: int | None = None
    difficulty: str = "medium"  # simple | medium | complex
    persona: str = ""
    preferences: dict[str, str | None] = field(default_factory=dict)


AGENT_TOOLS: tuple[dict[str, str], ...] = (
    {"name": "docSearch", "signature": "docSearch(query: str)", "description": "Top-k text snippets + docIDs from MRB."},
    {"name": "imgSearch", "signature": "imgSearch(imgPath: str)", "description": "Top-k visually similar images from MRB."},
    {"name": "getDocument", "signature": "getDocument(docID: str)", "description": "Full document text by docID."},
    {
        "name": "getRecommendRestaurant",
        "signature": "getRecommendRestaurant(city: str)",
        "description": "Top recommended restaurants in city (structured).",
    },
    {
        "name": "getRestaurantByName",
        "signature": "getRestaurantByName(city: str, name: str)",
        "description": "Restaurant metadata by name.",
    },
    {
        "name": "getRestaurantByFood",
        "signature": "getRestaurantByFood(city: str, food: str)",
        "description": "Restaurants serving a dish/cuisine.",
    },
)
