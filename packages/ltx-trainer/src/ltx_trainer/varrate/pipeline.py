"""VarRate evaluation pipeline and cards (arXiv:2607.15498)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.varrate.allocate import allocate_for_sequence
from ltx_trainer.varrate.baselines import PAPER_ANCHORS, benchmarks_bundle
from ltx_trainer.varrate.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, VarRateConfig


def framework_card(cfg: VarRateConfig | None = None) -> dict[str, Any]:
    c = cfg or VarRateConfig()
    return {
        "paper": {"arxiv": PAPER_ARXIV, "title": PAPER_TITLE, "url": PAPER_URL},
        "idea": (
            "Training-free variable-rank KV coding: water-fill per-token ranks from "
            "salience with a floor rmin; never evict tokens (unlike SnapKV/KVzip)."
        ),
        "kappa": c.kappa,
        "rmin": c.rmin,
        "r_max": c.r_max,
        "stride": c.stride,
        "future": "Fused decode / KVPress / Colibri cache_slot port is out of scope for this stub.",
    }


def knowledge_card() -> dict[str, Any]:
    return {
        "paper": {"arxiv": PAPER_ARXIV, "title": PAPER_TITLE, "url": PAPER_URL},
        "anchors": PAPER_ANCHORS,
        "limitations": [
            "CPU stub validates allocation math + paper anchors only.",
            "No glm.c / KVPress integration yet.",
            "Salience is a deterministic SnapKV-style proxy, not model attention.",
        ],
        "colibri": "Native analog: incremental prefill + cache_slot; VarRate would vary rank per token inside a slot.",
    }


def evaluation_demo(*, n_tokens: int = 256, seed: int = 0) -> dict[str, Any]:
    cfg = VarRateConfig()
    full = allocate_for_sequence(n_tokens, cfg=cfg, seed=seed)
    tight = allocate_for_sequence(
        n_tokens,
        budget=n_tokens * cfg.rmin,
        cfg=cfg,
        seed=seed,
    )
    return {
        "full_kappa": full,
        "floor_only": tight,
        "no_eviction": full["no_eviction"] and tight["no_eviction"],
        "mean_rank_above_rmin": full["allocation"]["mean_rank"] >= cfg.rmin,
        "budget_matches": abs(full["allocation"]["sum_ranks"] - full["allocation"]["budget"])
        <= max(2, n_tokens // 20),
        "paper_llama_avg": PAPER_ANCHORS["llama_varrate_avg"],
        "reuse_holds": abs(float(PAPER_ANCHORS["reuse_delta_varrate"])) < 1.0,
    }


def evaluation_smoke(cfg: VarRateConfig | None = None) -> dict[str, bool]:
    _ = cfg
    demo = evaluation_demo(n_tokens=128, seed=0)
    alloc = demo["full_kappa"]["allocation"]
    checks = {
        "no_eviction": demo["no_eviction"],
        "min_at_least_rmin": alloc["min_rank"] >= PAPER_ANCHORS["rmin"],
        "budget_ok": demo["budget_matches"],
        "mean_above_rmin": demo["mean_rank_above_rmin"],
        "kappa_anchor": abs(float(PAPER_ANCHORS["kappa"]) - 0.20) < 1e-9,
        "rmin_anchor": PAPER_ANCHORS["rmin"] == 16,
        "rmax_anchor": PAPER_ANCHORS["r_max"] == 1024,
        "llama_avg_anchor": abs(float(PAPER_ANCHORS["llama_varrate_avg"]) - 47.4) < 1e-9,
        "reuse_small_drop": demo["reuse_holds"],
        "prefill_overhead_anchor": abs(float(PAPER_ANCHORS["prefill_overhead_x"]) - 1.08) < 1e-9,
    }
    checks["all_pass"] = all(checks.values())
    return checks


def benchmarks_card() -> dict[str, Any]:
    return benchmarks_bundle()
