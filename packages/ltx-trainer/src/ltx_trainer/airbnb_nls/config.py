"""Configuration for Airbnb NL search cold-start (Wei et al., arXiv:2605.21812)."""

from __future__ import annotations

from dataclasses import dataclass

PROMPT_VARIANTS: tuple[str, ...] = ("seed_controlled", "seed_freeform", "variety")

PLATFORM_TERMS_BLOCKLIST: tuple[str, ...] = (
    "entire home",
    "private room",
    "superhost",
    "guest favorite",
    "badge",
    "airbnb",
    "listing",
    "instant book",
)

ATTRIBUTE_TYPES: tuple[str, ...] = (
    "amenity",
    "location",
    "property_type",
    "vibe",
    "rooms",
    "guest_context",
)


@dataclass
class AirbnbNLSConfig:
    """Defaults from paper Sec. 3–5."""

    seed_query_count: int = 500
    daily_synthetic_queries: int = 10_000
    seed_guided_mix_ratio: float = 0.80  # 80% seed-guided, 20% variety
    query_word_min: int = 3
    query_word_max: int = 8
    # Table 2 reference KL (length vs real)
    baseline_length_kl_vs_real: float = 4.95
    our_length_kl_vs_real: float = 0.66
    seed_length_kl_vs_real: float = 0.15
    # Table 3 attribute type KL vs real
    baseline_attr_type_kl: float = 0.13
    our_attr_type_kl: float = 0.04
    seed_attr_type_kl: float = 0.09
    # Table 5 retrieval accuracy (our approach)
    retrieval_accuracy_qwen3: float = 0.790
    retrieval_accuracy_baseline_max: float = 0.993
    # Table 6 ranking
    ranking_accuracy_baseline: float = 0.827
    ranking_accuracy_ours: float = 0.792
    # VJ human agreement (Sec. 4.2)
    virtual_judge_human_agreement: float = 0.87
