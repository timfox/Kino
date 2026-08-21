"""Framework card, conversion tables, benchmark anchors, CPU demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ucs_sfx.cascade import classify_tags
from ltx_trainer.ucs_sfx.conflict import resolve_file_categories
from ltx_trainer.ucs_sfx.config import UcsSfxConfig
from ltx_trainer.ucs_sfx.split import composite_key, split_distribution_correlation, stratified_split_indices


def framework_card(cfg: UcsSfxConfig | None = None) -> dict[str, Any]:
    c = cfg or UcsSfxConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "venue": c.venue,
        "ucs_version": c.ucs_version,
        "task": "sfx_dataset_unification",
        "pipeline": ["predefined", "subcategory", "category", "synonym", "conflict_resolution"],
        "envsound_ucs": f"{c.envsound_total:,} clips, {c.envsound_categories} categories",
        "github_tools": c.github_tools,
        "headline": headline_results(c),
    }


def headline_results(cfg: UcsSfxConfig | None = None) -> dict[str, Any]:
    c = cfg or UcsSfxConfig()
    return {
        "envsound_total": c.envsound_total,
        "envsound_categories": c.envsound_categories,
        "fsd50k_conversion_rate": c.fsd50k_classified_rate,
        "audioset_conversion_rate": c.audioset_classified_rate,
        "esc50_conversion_rate": c.esc50_classified_rate,
        "envsound_subcat_flat_f1": c.envsound_subcat_flat_f1,
        "envsound_subcat_hier_oracle_f1": c.envsound_subcat_hier_oracle_f1,
    }


def table1_conversion(cfg: UcsSfxConfig | None = None) -> list[dict[str, Any]]:
    c = cfg or UcsSfxConfig()
    return [
        {
            "dataset": "FSD50K",
            "total": c.fsd50k_total,
            "classified": c.fsd50k_total,
            "rate": c.fsd50k_classified_rate,
            "ambiguous_pct": c.fsd50k_ambiguous_pct,
        },
        {
            "dataset": "AudioSet",
            "total": c.audioset_total,
            "classified": c.audioset_classified,
            "rate": c.audioset_classified_rate,
            "ambiguous_pct": c.audioset_ambiguous_pct,
        },
        {
            "dataset": "ESC-50",
            "total": c.esc50_total,
            "classified": c.esc50_total,
            "rate": c.esc50_classified_rate,
            "ambiguous_pct": 0.0,
        },
    ]


def table2_envsound_sources(cfg: UcsSfxConfig | None = None) -> list[dict[str, Any]]:
    """Table 2 — top EnvSound-UCS category contributions (subset)."""
    c = cfg or UcsSfxConfig()
    return [
        {"category": "FOLEY", "esc": 120, "fsd": 6814, "as": 1073, "total": 8007},
        {"category": "ANIMALS", "esc": 240, "fsd": 3641, "as": 1622, "total": 5503},
        {"category": "VEHICLES", "esc": 0, "fsd": 2367, "as": 1327, "total": 3694},
        {"category": "HUMAN", "esc": 200, "fsd": 1444, "as": 1251, "total": 2895},
        {"category": "WATER", "esc": 160, "fsd": 1336, "as": 1040, "total": 2536},
        {"category": "ALL", "esc": c.envsound_esc50, "fsd": c.envsound_fsd50k, "as": c.envsound_audioset, "total": c.envsound_total},
    ]


def table3_benchmark_original(cfg: UcsSfxConfig | None = None) -> list[dict[str, Any]]:
    c = cfg or UcsSfxConfig()
    return [
        {"dataset": "FSD50K", "cat_f1": c.fsd50k_cat_f1, "subcat_hier_oracle_f1": 0.86},
        {"dataset": "AudioSet", "cat_f1": c.audioset_cat_f1, "subcat_hier_oracle_f1": 0.74},
        {"dataset": "ESC-50", "cat_f1": c.esc50_cat_f1, "subcat_hier_oracle_f1": 0.95},
    ]


def table4_envsound_benchmark(cfg: UcsSfxConfig | None = None) -> dict[str, Any]:
    c = cfg or UcsSfxConfig()
    return {
        "categories": c.envsound_categories,
        "cat_f1": c.envsound_cat_f1,
        "subcat_flat_f1": c.envsound_subcat_flat_f1,
        "subcat_hier_oracle_f1": c.envsound_subcat_hier_oracle_f1,
        "cross_source_subcat_flat_f1": {
            "FSD-env": 0.62,
            "AS-env": 0.46,
            "ESC-50": 0.92,
        },
    }


def benchmarks_bundle(cfg: UcsSfxConfig | None = None) -> dict[str, Any]:
    c = cfg or UcsSfxConfig()
    return {
        "table1_conversion": table1_conversion(c),
        "table2_envsound": table2_envsound_sources(c),
        "table3_original": table3_benchmark_original(c),
        "table4_envsound": table4_envsound_benchmark(c),
        "ucs": {
            "version": c.ucs_version,
            "categories": c.ucs_categories,
            "subcategories": c.ucs_subcategories,
            "synonyms": c.ucs_synonyms,
        },
    }


def pipeline_demo(seed: int = 42, cfg: UcsSfxConfig | None = None) -> dict[str, Any]:
    c = cfg or UcsSfxConfig()
    mapping = {
        "dog bark": ("ANIMALS", "DOG"),
        "gunshot and gunfire": ("GUNS", "GUNSHOT"),
        "coin dropping": ("FOLEY", None),
        "singing": ("VOICES", None),
        "human voice": ("VOICES", None),
    }
    synonyms = {"bark": ("ANIMALS", "DOG")}
    categories = {"FOLEY", "VOICES", "DESIGNED"}
    subcategories = {"dog": "ANIMALS"}

    tags_conflict = ["coin dropping", "singing", "human voice"]
    matches_conflict = classify_tags(
        tags_conflict,
        mapping=mapping,
        subcategories=subcategories,
        categories=categories,
        synonyms=synonyms,
    )
    cat_c, sub_c, amb_c = resolve_file_categories(matches_conflict)

    tags_easy = ["dog bark"]
    matches_easy = classify_tags(tags_easy, mapping=mapping, synonyms=synonyms, subcategories=subcategories)
    cat_e, _, amb_e = resolve_file_categories(matches_easy)

    keys = [composite_key("FOLEY", "COIN"), composite_key("ANIMALS", "DOG")] * 50
    keys += [composite_key("WATER", None)] * 50
    split = stratified_split_indices(keys, seed=seed)
    corr = split_distribution_correlation(keys, split["train"], split["test"])

    t1 = table1_conversion(c)
    return {
        "conflict_category": cat_c,
        "conflict_ambiguous": amb_c,
        "easy_category": cat_e,
        "easy_ambiguous": amb_e,
        "split_corr": round(corr, 4),
        "split_passes_threshold": corr > c.split_corr_min,
        "envsound_total": c.envsound_total,
        "fsd50k_rate": t1[0]["rate"],
        "audioset_rate": t1[1]["rate"],
        "envsound_subcat_flat_f1": c.envsound_subcat_flat_f1,
    }


def evaluation_demo(seed: int = 42, cfg: UcsSfxConfig | None = None) -> dict[str, Any]:
    return pipeline_demo(seed=seed, cfg=cfg)
