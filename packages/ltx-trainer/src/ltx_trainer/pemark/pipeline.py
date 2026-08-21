"""PEMark framework card, paper tables, and smoke demos (arXiv:2605.21865)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pemark.attacks import deletion_attack, insertion_attack, tamper_attack_values_stub, watermark_similarity
from ltx_trainer.pemark.config import PEMarkConfig
from ltx_trainer.pemark.layout import LIMITATIONS
from ltx_trainer.pemark.position_encoding import (
    embed_watermark_into_keys,
    extract_with_group_voting,
    int_to_watermark_bits,
    min_threshold_T_for_bits,
    watermark_bits_to_int,
)


def framework_card(cfg: PEMarkConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PEMarkConfig()
    return {
        "name": "PEMark",
        "paper": cfg.paper_arxiv,
        "idea": (
            "Distortion-free API response watermarking via proxy gateway and position encoding: "
            "embed watermark by reordering JSON/XML keys (permutation redundancy) without changing any values; "
            "extract by inverting Lehmer code and majority voting across groups."
        ),
        "channel": "key_ordering_redundancy",
        "proxy_gateway": {"deployed_between": ["server", "client"], "stack": cfg.gateway_stack},
        "encoding": {"lehmer_code": True, "factorial_series": True, "capacity_condition": "2^L <= T!"},
        "defaults": cfg.__dict__,
    }


def table_i_method_comparison() -> list[dict[str, Any]]:
    """Table I — high-level comparison."""
    return [
        {
            "method": "Modify structured data",
            "requires_business_code_change": True,
            "requires_third_party_library": False,
            "affects_business_usability": True,
        },
        {
            "method": "Modify semi-structured data",
            "requires_business_code_change": True,
            "requires_third_party_library": False,
            "affects_business_usability": True,
        },
        {
            "method": "Zero-Watermark",
            "requires_business_code_change": False,
            "requires_third_party_library": True,
            "affects_business_usability": False,
        },
        {
            "method": "Position-Encoding (PEMark)",
            "requires_business_code_change": False,
            "requires_third_party_library": False,
            "affects_business_usability": False,
        },
    ]


def table_ii_dataset_configs() -> list[dict[str, Any]]:
    """Table II — dataset configs (constructed JSON)."""
    return [
        {"id": 1, "keys_range": "5–25", "max_depth": 1, "nest_prob": 0.0},
        {"id": 2, "keys_range": "5–25", "max_depth": 3, "nest_prob": 0.3},
        {"id": 3, "keys_range": "5–25", "max_depth": 3, "nest_prob": 0.7},
        {"id": 4, "keys_range": "50–250", "max_depth": 1, "nest_prob": 0.0},
        {"id": 5, "keys_range": "50–250", "max_depth": 3, "nest_prob": 0.3},
        {"id": 6, "keys_range": "50–250", "max_depth": 3, "nest_prob": 0.7},
        {"id": 7, "keys_range": "50", "max_depth": 3, "nest_prob": 0.3},
        {"id": 8, "keys_range": "100", "max_depth": 3, "nest_prob": 0.3},
        {"id": 9, "keys_range": "200", "max_depth": 3, "nest_prob": 0.3},
    ]


def fig5_threshold_vs_length_points() -> list[dict[str, int]]:
    """Fig. 5 — sample points; exact curve depends on factorial growth."""
    # Use the capacity condition 2^L <= T! to compute representative T values.
    points = []
    for L in (32, 64, 96, 128):
        points.append({"L_bits": L, "T_threshold": min_threshold_T_for_bits(L)})
    return points


def table_iii_api_latency() -> list[dict[str, float | str]]:
    """Table III — API response time overhead (ms)."""
    return [
        {"api": "DeepSeek", "original_ms": 543.24, "with_pemark_ms": 563.19, "overhead_ms": 19.95},
        {"api": "OpenAI", "original_ms": 933.92, "with_pemark_ms": 952.47, "overhead_ms": 18.55},
        {"api": "GitHub", "original_ms": 702.74, "with_pemark_ms": 723.32, "overhead_ms": 20.58},
    ]


def robustness_excerpt() -> dict[str, Any]:
    """Paper excerpt summary from Fig. 7 text."""
    return {
        "tamper_similarity_all": 100.0,
        "insert_similarity_all": 100.0,
        "delete_similarity_below_15pct_attack": ">=94%",
        "delete_similarity_at_50pct_attack": "49.23%–60.45%",
        "embedding_overhead_ms_max": 0.65,
    }


def pipeline_demo(cfg: PEMarkConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PEMarkConfig()
    L = cfg.watermark_bits_L
    T = cfg.threshold_T
    # Build a deterministic key list with multiple groups.
    keys = [f"k{i:03d}" for i in range(T * 4 + 3)]
    wm_bits = int_to_watermark_bits(0x1234_5678_9ABC_DEF0 % (1 << L), L)
    watermarked = embed_watermark_into_keys(keys, watermark_bits=wm_bits, T=T)
    recovered = extract_with_group_voting(watermarked, L=L, T=T)

    # Attacks (toy): delete changes ordering info by removing keys; tamper shouldn't matter; insert shouldn't matter.
    deleted = deletion_attack(watermarked, intensity=0.2, seed=1)
    tampered = tamper_attack_values_stub(watermarked, intensity=0.5)
    inserted = insertion_attack(watermarked, intensity=0.5, seed=2)

    rec_del = extract_with_group_voting(deleted, L=L, T=T)
    rec_tam = extract_with_group_voting(tampered, L=L, T=T)
    rec_ins = extract_with_group_voting(inserted, L=L, T=T)

    return {
        "L_bits": L,
        "T_threshold": T,
        "groups": len(keys) // T,
        "watermark_int": watermark_bits_to_int(wm_bits),
        "recovered_exact": recovered == wm_bits,
        "similarity_delete_20pct_toy": round(watermark_similarity(wm_bits, rec_del), 2),
        "similarity_tamper_50pct_toy": round(watermark_similarity(wm_bits, rec_tam), 2),
        "similarity_insert_50pct_toy": round(watermark_similarity(wm_bits, rec_ins), 2),
    }


def evaluation_demo(cfg: PEMarkConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PEMarkConfig()
    pts = fig5_threshold_vs_length_points()
    t64 = next(p for p in pts if p["L_bits"] == 64)["T_threshold"]
    return {
        "framework": framework_card(cfg),
        "pipeline": pipeline_demo(cfg),
        "limitations": list(LIMITATIONS),
        "paper_tables": {
            "table_i_comparison": table_i_method_comparison(),
            "table_ii_datasets": table_ii_dataset_configs(),
            "table_iii_api_latency": table_iii_api_latency(),
            "fig5_threshold_vs_length": pts,
            "robustness_excerpt": robustness_excerpt(),
        },
        "capacity_check_L64_T": {"L_bits": 64, "T_threshold": t64, "paper_claim_T": 21, "matches": t64 == 21},
    }

