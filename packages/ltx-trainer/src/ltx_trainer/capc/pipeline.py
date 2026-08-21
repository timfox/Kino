"""CAPC evaluation pipeline and cards (arXiv:2607.15516)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.capc.baselines import PAPER_ANCHORS, RHO_CROSS_TABLE, benchmarks_bundle
from ltx_trainer.capc.boundary import (
    AdaptiveCacheBoundary,
    adaptive_vs_naive_savings,
    synthetic_version_drift,
)
from ltx_trainer.capc.compress import compress_query_agnostic
from ltx_trainer.capc.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, CapcConfig
from ltx_trainer.capc.cost_model import rho_cross, strategy_costs, tier_preserving_rmax


def framework_card(cfg: CapcConfig | None = None) -> dict[str, Any]:
    c = cfg or CapcConfig()
    return {
        "paper": {"arxiv": PAPER_ARXIV, "title": PAPER_TITLE, "url": PAPER_URL},
        "idea": (
            "Query-agnostic prompt compression + explicit caching with a "
            "tier-preserving ratio bound so the compressed prefix stays in the "
            "persistent cache tier (≥ ~3500 tokens on Sonnet 4.6)."
        ),
        "hot_tier_tokens": c.hot_tier_tokens,
        "rho_hot": c.rho_hot,
        "pricing": {"pin": c.pin, "cw": c.cw, "cr": c.cr, "pout": c.pout},
        "strategies": ("A_vanilla", "B_cache_only", "C_query_aware", "D_capc"),
        "colibri": (
            "MemPalace WAKE.md / Anamnesis: compress wake query-agnostically; "
            "keep search in the per-turn user message (static first, dynamic last)."
        ),
    }


def knowledge_card() -> dict[str, Any]:
    return {
        "paper": {"arxiv": PAPER_ARXIV, "title": PAPER_TITLE, "url": PAPER_URL},
        "anchors": PAPER_ANCHORS,
        "rho_cross_table": RHO_CROSS_TABLE,
        "limitations": [
            "CPU stub uses sentence-prefix truncation, not LLMLingua / learned scorers.",
            "pin/cw/cr/pout are Sonnet-class parametric prices for ranking only.",
            "LongBench/τ-bench numbers are paper anchors, not a live reproduction.",
            "Colibri maps cache → local KV prefix reuse, not Anthropic cache_control.",
        ],
        "gopex": {
            "wake_env": "GOPEX_CAPC_WAKE=1",
            "wake_budget": "COLI_MEMORY_WAKE_MAX_CHARS",
            "aider_tip": "Static first, dynamic last; attach only files you edit; do not mutate large tool schemas per turn.",
        },
    }


def evaluation_demo(
    *,
    doc_tokens: int = 14000,
    prompt_tokens: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    cfg = CapcConfig()
    r_max = tier_preserving_rmax(doc_tokens, cfg)
    # Prefer mid ratio that stays tier-preserving (paper sweet spot often r=3).
    r = float(min(max(2, r_max), 3) if r_max >= 2 else max(1, r_max))
    costs = strategy_costs(
        prompt_tokens=prompt_tokens,
        doc_tokens=doc_tokens,
        r=r,
        cfg=cfg,
        n_calls=10,
    )
    sentences = [
        f"Drawer {i}: decision about component {i % 7} and follow-up action {i}."
        for i in range(40)
    ]
    doc = " ".join(sentences)
    compressed = compress_query_agnostic(doc, ratio=r, cfg=cfg)

    versions = synthetic_version_drift(base_tokens=4000, k=12, mutation_rate=0.45, seed=seed)
    boundary = AdaptiveCacheBoundary(cfg=cfg)
    for v in versions:
        boundary.observe(v)
    prefix = boundary.build_cache_prefix()
    savings = adaptive_vs_naive_savings(versions, n_queries=10, cfg=cfg)

    # §7.2 mutated-fraction sign flip demo (EA-like small mf vs full bust).
    full_bust = strategy_costs(
        prompt_tokens=prompt_tokens, doc_tokens=doc_tokens, r=3.0, cfg=cfg, mutated_frac=1.0
    )
    tools_stable = strategy_costs(
        prompt_tokens=prompt_tokens, doc_tokens=doc_tokens, r=3.0, cfg=cfg, mutated_frac=0.10
    )

    return {
        "r_max": r_max,
        "r_used": r,
        "rho_cross": rho_cross(r, cfg),
        "strategy_costs": costs,
        "compress": {
            "original_chars": compressed["original_chars"],
            "compressed_chars": compressed["compressed_chars"],
            "ratio": compressed["ratio"],
            "method": compressed["method"],
            "truncated": compressed["truncated"],
        },
        "boundary": {
            "prefix_chars": len(prefix),
            "composition": boundary.composition(),
            "positive_savings": savings["positive_savings"],
            "savings_frac": savings["savings_frac"],
        },
        "mutated_frac_demo": {
            "full_bust_cheapest": full_bust["cheapest"],
            "tools_stable_c_cost": tools_stable["costs"]["C_query_aware"],
            "full_bust_c_cost": full_bust["costs"]["C_query_aware"],
            "sign_flip": tools_stable["costs"]["C_query_aware"] < full_bust["costs"]["C_query_aware"],
        },
        "capc_cheapest": costs["capc_wins"],
        "longbench_dominance": PAPER_ANCHORS["longbench_dominance"],
    }


def benchmarks_card() -> dict[str, Any]:
    return benchmarks_bundle()


def evaluation_smoke(cfg: CapcConfig | None = None) -> dict[str, bool]:
    c = cfg or CapcConfig()
    demo = evaluation_demo(doc_tokens=14000, prompt_tokens=200, seed=0)
    costs = demo["strategy_costs"]["costs"]
    # Paper invariants on the 12,191-token doc (Table 7 / §5.2).
    rmax_12k = tier_preserving_rmax(12191, c)
    rho6 = rho_cross(6.0, c)

    # Large-doc mid-r: CAPC should be cheapest.
    large = strategy_costs(prompt_tokens=200, doc_tokens=24576, r=3.0, cfg=c, n_calls=10)

    versions = synthetic_version_drift(base_tokens=3000, k=10, mutation_rate=0.05, seed=1)
    boundary = AdaptiveCacheBoundary(cfg=c)
    for v in versions:
        boundary.observe(v)
    prefix = boundary.build_cache_prefix()

    checks = {
        "r_max_positive": demo["r_max"] >= 1,
        "r_safe_max_12191": rmax_12k == 3,
        "capc_cheapest": demo["capc_cheapest"],
        "capc_cheaper_than_vanilla": costs["D_capc"] < costs["A_vanilla"],
        "capc_cheaper_than_query_aware": costs["D_capc"] <= costs["C_query_aware"],
        "large_doc_capc_wins": large["capc_wins"],
        "rho_cross_r6": rho6 > 0.94,
        "rho_cross_table_r6": abs(float(RHO_CROSS_TABLE[6]) - 0.942) < 1e-6,
        "rho_in_unit": 0.0 <= demo["rho_cross"] <= 1.0,
        "compress_shrinks": demo["compress"]["compressed_chars"] < demo["compress"]["original_chars"],
        "boundary_prefix_nonempty": len(prefix) > 0,
        "boundary_savings": bool(demo["boundary"]["positive_savings"]),
        "hot_tier_anchor": PAPER_ANCHORS["hot_tier_tokens"] == 3500,
        "rho_hot_anchor": abs(float(PAPER_ANCHORS["rho_hot"]) - 0.83) < 1e-9,
        "longbench_16_16": PAPER_ANCHORS["longbench_dominance"] == 16,
        "tau_reward_tie": PAPER_ANCHORS["tau_reward_capc"] == PAPER_ANCHORS["tau_reward_vanilla"],
    }
    checks["all_pass"] = all(checks.values())
    return checks
