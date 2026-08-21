"""Framework card, paper tables, and generation demos."""

from __future__ import annotations

from typing import Any

from ltx_trainer.airbnb_nls.config import PROMPT_VARIANTS, AirbnbNLSConfig
from ltx_trainer.airbnb_nls.generation import (
    contrastive_generate_step,
    demo_listing_pair,
    generation_modes_catalog,
)
from ltx_trainer.airbnb_nls.labels import contrastive_label, llm_self_preference_accuracy_table
from ltx_trainer.airbnb_nls.metrics import kl_divergence, length_stats, word_count_distribution


def framework_card(cfg: AirbnbNLSConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AirbnbNLSConfig()
    return {
        "name": "AirbnbNLSColdStart",
        "paper": "arXiv:2605.21812",
        "title": "Bridging the Cold-Start Gap: LLM-Powered Synthetic Data for NL Search at Airbnb",
        "problem": ["no_real_queries", "no_relevance_labels"],
        "solutions": ["contrastive_query_generation", "virtual_judge_labeling"],
        "prompt_variants": list(PROMPT_VARIANTS),
        "seed_query_count": cfg.seed_query_count,
        "daily_synthetic_queries": cfg.daily_synthetic_queries,
        "production_mix": {"seed_guided": cfg.seed_guided_mix_ratio, "variety": 1 - cfg.seed_guided_mix_ratio},
        "focus": "topicality",
        "future_work": "bookability_ndcg",
    }


def table_query_length() -> dict[str, dict[str, float]]:
    """Table 2 — query length statistics."""
    return {
        "baseline": {
            "count": 4912,
            "length_mean": 13.6,
            "length_median": 15,
            "length_std": 2.0,
            "pct_3_8_words": 2.4,
            "pct_short_1_2": 0.0,
            "pct_long_9_plus": 97.6,
            "kl_vs_real": 4.95,
            "kl_vs_seed": 3.67,
        },
        "our_approach": {
            "count": 15742,
            "length_mean": 6.0,
            "length_median": 6,
            "length_std": 2.5,
            "pct_3_8_words": 84.0,
            "pct_short_1_2": 6.0,
            "pct_long_9_plus": 10.0,
            "kl_vs_real": 0.66,
            "kl_vs_seed": 0.38,
        },
        "seed": {
            "count": 500,
            "length_mean": 6.6,
            "length_median": 5,
            "length_std": 4.1,
            "pct_3_8_words": 56.8,
            "pct_short_1_2": 15.6,
            "pct_long_9_plus": 27.6,
            "kl_vs_real": 0.15,
        },
        "real": {
            "count": 1546,
            "length_mean": 5.0,
            "length_median": 4,
            "length_std": 3.96,
            "pct_3_8_words": 49.8,
            "pct_short_1_2": 32.9,
            "pct_long_9_plus": 17.3,
        },
    }


def table_attribute_type_kl() -> dict[str, float]:
    """Table 3 — attribute type KL vs real."""
    return {"baseline": 0.13, "our_approach": 0.04, "seed": 0.09}


def table_retrieval_accuracy() -> dict[str, dict[str, float]]:
    """Table 5 — embedding pairwise accuracy."""
    return {
        "sbert_minilm": {"baseline": 0.887, "our_approach": 0.777},
        "qwen3": {"baseline": 0.967, "our_approach": 0.790},
        "openai_text_large": {"baseline": 0.993, "our_approach": 0.763},
    }


def table_ranking_accuracy() -> dict[str, dict[str, float]]:
    """Table 6 — first-pass ranking training."""
    return {
        "baseline": {"dataset_size": 1200, "accuracy": 0.827, "loss": 0.623},
        "our_approach": {"dataset_size": 1000, "accuracy": 0.792, "loss": 0.657},
    }


def table_prompt_variant_kl() -> dict[str, dict[str, float]]:
    """Table 8 — KL by prompt variant."""
    return {
        "seed_controlled": {"length_vs_seed": 0.07, "length_vs_real": 0.18, "attr_cnt_vs_real": 1.00, "attr_type_vs_real": 0.11},
        "seed_freeform": {"length_vs_seed": 0.29, "length_vs_real": 0.22, "attr_cnt_vs_real": 3.50, "attr_type_vs_real": 0.21},
        "variety": {"length_vs_seed": 0.26, "length_vs_real": 0.13, "attr_cnt_vs_real": 8.69, "attr_type_vs_real": 0.26},
    }


def demo_queries_for_kl() -> dict[str, list[str]]:
    """Synthetic query lists matching Table 2 length profiles (for KL smoke tests)."""
    return {
        "real": ["pet friendly cabin", "pool near beach", "cozy cabin", "wifi downtown"] * 50,
        "our_approach": ["pool near beach", "romantic escape wifi", "cabin ski resort"] * 50,
        "baseline": [
            "I am seeking accommodations with swimming pool facilities near the beach area"
        ]
        * 50,
    }


def generation_step_demo(cfg: AirbnbNLSConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AirbnbNLSConfig()
    pos, neg = demo_listing_pair()
    seeds = [
        "family friendly place with pool and big kitchen safe neighborhood",
        "romantic getaway near beach",
    ]
    triplet = contrastive_generate_step(
        seed_queries=seeds,
        positive=pos,
        negative=neg,
        prompt_variant="seed_controlled",
    )
    pos_lbl, neg_lbl = contrastive_label(triplet)
    demos = demo_queries_for_kl()
    p_real = word_count_distribution(demos["real"])
    p_ours = word_count_distribution(demos["our_approach"])
    return {
        "query": triplet.query,
        "prompt_variant": triplet.prompt_variant,
        "positive_topical": pos_lbl.is_more_relevant,
        "negative_topical": neg_lbl.is_more_relevant,
        "kl_ours_vs_real": kl_divergence(p_ours, p_real),
        "length_stats_ours": length_stats(demos["our_approach"]),
        "modes": generation_modes_catalog(cfg),
    }
