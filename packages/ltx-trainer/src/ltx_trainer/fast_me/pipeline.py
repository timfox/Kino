"""FAST-ME framework card, paper tables, and smoke demos (arXiv:2605.23428)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fast_me.blend import blended_cost, should_stop_fast_me, stopping_boundary
from ltx_trainer.fast_me.config import FastMEConfig
from ltx_trainer.fast_me.layout import LIMITATIONS
from ltx_trainer.fast_me.mock import toy_attention_background, toy_attention_high_motion, toy_sad_sequence
from ltx_trainer.fast_me.ost import empirical_cdf, exponential_stop_threshold, ost_stop_index
from ltx_trainer.fast_me.sad import sad_block, toy_blocks


def framework_card(cfg: FastMEConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FastMEConfig()
    return {
        "name": "FAST-ME",
        "paper": cfg.paper_arxiv,
        "idea": (
            "Foundation-aware adaptive stopping for block motion estimation: OST on SAD "
            "fused with ViT/SAM/CLIP semantic attention for early stop in redundant regions."
        ),
        "stages": [
            "Adaptive ME (OST on empirical/exponential SAD CDF)",
            "Semantic attention Ak per macroblock (foundation models)",
            "Blended cost Ȳ_k = α·Y_k + (1-α)·(1-A_k)",
            "Dynamic threshold T_k = -log(δ_0(1-A_k))/θ",
        ],
        "defaults": {
            "alpha": cfg.alpha,
            "delta0": cfg.delta0,
            "theta": cfg.theta,
            "block_size": cfg.default_block_size,
            "search_range_p": cfg.default_search_range,
        },
        "foundation_models": list(cfg.fm_backends),
        "baselines": ["Full Search", "Diamond Search", "Three-Step Search", "Adaptive ME"],
        "datasets": ["Foreman", "BridgeFar (DERF)", "Four People", "Big Buck Bunny"],
    }


def table_alpha_tradeoff() -> dict[str, dict[str, float | int]]:
    """Table II — impact of blending factor α."""
    return {
        "0.3": {"mean_sad": 11214, "scs_pct": 76.5, "comparisons": 980},
        "0.5": {"mean_sad": 9812, "scs_pct": 72.4, "comparisons": 1150},
        "0.7": {"mean_sad": 9012, "scs_pct": 69.7, "comparisons": 1320},
        "0.9": {"mean_sad": 8856, "scs_pct": 61.0, "comparisons": 1670},
    }


def table_foreman_me() -> dict[str, dict[str, float]]:
    """Table III — Foreman sequence (time, avg SAD, comparisons)."""
    return {
        "FS": {"time_s": 0.0709, "avg_sad": 4.6036e5, "comparisons": 80896},
        "DS": {"time_s": 2.1754, "avg_sad": 1.5527e4, "comparisons": 4391},
        "TSS": {"time_s": 4.8145, "avg_sad": 4.1761e4, "comparisons": 10102},
        "Adaptive_ME_T50": {"time_s": 0.0130, "avg_sad": 1.0094e4, "comparisons": 1495},
        "FAST_ME": {"time_s": 0.0013, "avg_sad": 9.012e3, "comparisons": 1320},
    }


def table_foreman_psnr_scs() -> dict[str, dict[str, float]]:
    """Table IV — PSNR, comparisons, semantic coverage score (SCS)."""
    return {
        "Full_Search": {"psnr": 27.89, "comparisons": 80896, "scs_pct": 45.6},
        "TSS": {"psnr": 24.82, "comparisons": 10102, "scs_pct": 50.3},
        "Diamond_Search": {"psnr": 26.89, "comparisons": 4391, "scs_pct": 52.1},
        "Adaptive_ME": {"psnr": 25.43, "comparisons": 1495, "scs_pct": 55.2},
        "FAST_ME": {"psnr": 26.12, "comparisons": 1320, "scs_pct": 69.7},
    }


def table_multi_sequence() -> dict[str, dict[str, float | int | str]]:
    """Table V excerpt — time, SAD, comparisons across sequences."""
    return {
        "Foreman_352x288": {
            "FS_time": 1.04,
            "FAST_ME_time": 0.68,
            "Ad_ME_comparisons": 1536,
            "FAST_ME_comparisons": 2300,
        },
        "BridgeFar_352x288": {
            "FS_time": 1.03,
            "FAST_ME_time": 0.42,
            "Ad_ME_comparisons": 864,
            "FAST_ME_comparisons": 1296,
        },
        "Big_Bunny_1920x1080": {
            "FS_time": 1.02,
            "FAST_ME_time": 0.27,
            "Ad_ME_comparisons": 528,
            "FAST_ME_comparisons": 792,
        },
    }


def pipeline_demo(cfg: FastMEConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FastMEConfig()
    bc, br = toy_blocks()
    sad = sad_block(bc, br)
    seq = toy_sad_sequence()
    stop_k = ost_stop_index(seq, delta=cfg.delta0)
    y_thr = exponential_stop_threshold(delta=cfg.delta0, theta=cfg.theta)
    attn_hi = toy_attention_high_motion()
    attn_lo = toy_attention_background()
    return {
        "toy_sad_2x2": sad,
        "ost_stop_index": stop_k,
        "exponential_threshold": y_thr,
        "empirical_cdf_at_min": empirical_cdf(seq, min(seq)),
        "blended_cost_salient": blended_cost(sad, attn_hi, alpha=cfg.alpha),
        "blended_cost_background": blended_cost(sad, attn_lo, alpha=cfg.alpha),
        "stopping_boundary_salient": stopping_boundary(attn_hi, delta0=cfg.delta0, theta=cfg.theta),
        "stopping_boundary_background": stopping_boundary(attn_lo, delta0=cfg.delta0, theta=cfg.theta),
        "fast_me_stop_salient": should_stop_fast_me(
            sad, attn_hi, best_sad=float("inf"), cfg=cfg
        ),
    }


def evaluation_demo(cfg: FastMEConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FastMEConfig()
    foreman = table_foreman_me()
    fs_comp = foreman["FS"]["comparisons"]
    fast_comp = foreman["FAST_ME"]["comparisons"]
    reduction_pct = (1.0 - fast_comp / fs_comp) * 100.0
    return {
        "framework": framework_card(cfg),
        "pipeline": pipeline_demo(cfg),
        "limitations": list(LIMITATIONS),
        "computation_reduction_vs_fs_pct": round(reduction_pct, 2),
        "paper_tables": {
            "alpha_tradeoff": table_alpha_tradeoff(),
            "foreman_me": foreman,
            "foreman_psnr_scs": table_foreman_psnr_scs(),
            "multi_sequence": table_multi_sequence(),
        },
    }
