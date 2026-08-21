"""FORTE framework card, tables, and demos (arXiv:2606.05812)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.forte.config import ForteConfig
from ltx_trainer.forte.layout import LIMITATIONS
from ltx_trainer.forte.mock import (
    evaluation_smoke,
    run_alignment_smoke,
    run_fol_refinement_smoke,
    run_pipeline_smoke,
    run_rerank_smoke,
)
from ltx_trainer.forte.prompts import prompt_bundle


def framework_card(cfg: ForteConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ForteConfig()
    return {
        "name": "FORTE — FOL-guided Optimal Refinement for Text-audio rEtrieval",
        "paper": cfg.paper_arxiv,
        "authors": "Arghya Pal, Sailaja Rajanala (Monash University)",
        "task": "Text-to-audio retrieval with FOL query refinement, parameter-efficient alignment, predicate re-ranking",
        "idea": (
            "Transform queries to first-order logic, refine via constrained best-first search preserving "
            "invariant predicates, align audio with a lightweight MLP projection, re-rank by predicate overlap."
        ),
        "stages": [
            "Stage 1 — FOL-guided query refinement (beam search over o_attr/o_rel/o_neg)",
            "Stage 2 — Parameter-efficient cross-modal alignment (frozen encoders + h_psi MLP)",
            "Stage 3 — Predicate-aware re-ranking (caption FOL vs phi* overlap)",
        ],
        "backbones": list(cfg.backbones),
        "datasets": list(cfg.datasets),
        "fol_parser": cfg.fol_parser,
        "elaboration_llm": cfg.elaboration_llm,
        "vocab_predicates": cfg.vocab_predicates,
        "hyperparameters": {
            "lambda": cfg.lambda_neg,
            "beta": cfg.beta_pivot,
            "mu": cfg.mu_logic,
            "tau": cfg.feasibility_tau,
            "alpha": cfg.alpha_rerank,
            "beam_offline": cfg.beam_offline,
            "beam_online": cfg.beam_online,
        },
        "defaults": cfg.__dict__,
    }


def table_i_main_results() -> list[dict[str, str | float | None]]:
    """Table 1 — Text-to-audio retrieval on AudioCaps and Clotho."""
    rows = [
        ("CLAP", "Frozen", "AC", None, 33.9, 72.0, 83.9, None, None, 14.4, 36.0, 49.9, None),
        ("CLAP", "FORTE†", "AC, Cl", 51.3, 36.4, 73.8, 85.3, 97.6, 29.8, 18.9, 43.7, 57.2, 85.3),
        ("LAION-CLAP", "D – CNN+HTSAT", "AC, Cl, WT5K", 49.45, 34.69, 70.22, 82.0, 97.28, 27.12, 16.75, 41.09, 54.07, 83.79),
        ("LAION-CLAP", "FORTE†", "AC, Cl, WT5K", 53.8, 38.2, 75.1, 86.8, 98.1, 32.5, 20.4, 46.3, 59.8, 87.2),
        ("Pengi", "Frozen", None, None, None, None, None, None, None, 9.4, 26.1, 36.7, None),
        ("Pengi", "FORTE†", "AC, Cl", 38.9, 25.7, 57.4, 72.3, 93.4, 19.4, 12.8, 31.9, 44.2, 74.6),
    ]
    cols = (
        "backbone",
        "method",
        "data",
        "map_ac",
        "r1_ac",
        "r5_ac",
        "r10_ac",
        "r50_ac",
        "map_cl",
        "r1_cl",
        "r5_cl",
        "r10_cl",
        "r50_cl",
    )
    out = []
    for row in rows:
        d = dict(zip(cols, row, strict=True))
        if d["map_ac"] is None and d["r1_ac"] is None:
            for k in ("map_ac", "r1_ac", "r5_ac", "r10_ac", "r50_ac"):
                d.pop(k)
        out.append(d)
    return out


def table_ii_alignment_loss() -> list[dict[str, float | str]]:
    """Table 2 — Effect of alignment loss on Clotho (LAION-CLAP)."""
    return [
        {"loss": "Binary contrastive (BCE)", "r1": 13.6, "map10": 25.9},
        {"loss": "Margin ranking loss", "r1": 17.1, "map10": 28.7},
        {"loss": "Triplet loss", "r1": 19.4, "map10": 31.2},
        {"loss": "FORTE (align only)", "r1": 19.7, "map10": 31.9},
        {"loss": "FORTE", "r1": 20.4, "map10": 32.5},
    ]


def table_iii_stage_ablation() -> list[dict[str, float | str | bool]]:
    """Table 3 — Stage-wise ablation on Clotho (LAION-CLAP)."""
    return [
        {"s1": False, "s2": False, "s3": False, "r1": 16.75, "r5": 41.09, "r10": 54.07, "map10": 27.12, "delta_r1": 0.0},
        {"s1": True, "s2": False, "s3": False, "r1": 18.3, "r5": 43.1, "r10": 56.2, "map10": 29.0, "delta_r1": 1.55},
        {"s1": False, "s2": True, "s3": False, "r1": 18.0, "r5": 42.7, "r10": 55.9, "map10": 28.7, "delta_r1": 1.25},
        {"s1": False, "s2": False, "s3": True, "r1": 17.5, "r5": 41.9, "r10": 54.8, "map10": 27.9, "delta_r1": 0.75},
        {"s1": True, "s2": True, "s3": False, "r1": 19.6, "r5": 45.1, "r10": 58.4, "map10": 31.1, "delta_r1": 2.85},
        {"s1": True, "s2": False, "s3": True, "r1": 19.1, "r5": 44.3, "r10": 57.6, "map10": 30.4, "delta_r1": 2.35},
        {"s1": False, "s2": True, "s3": True, "r1": 18.8, "r5": 43.9, "r10": 57.1, "map10": 30.0, "delta_r1": 2.05},
        {"s1": True, "s2": True, "s3": True, "r1": 20.4, "r5": 46.3, "r10": 59.8, "map10": 32.5, "delta_r1": 3.65},
    ]


def table_iv_anchor_bank() -> list[dict[str, float | str | None]]:
    """Table 4 — Anchor source comparison on Clotho."""
    return [
        {"anchor": "No anchor (pivot only)", "l2_dist": None, "r1": 18.3},
        {"anchor": "R0 top-1 (circular)†", "l2_dist": 0.81, "r1": 18.0},
        {"anchor": "Anchor bank B (ours)", "l2_dist": 0.54, "r1": 19.8},
        {"anchor": "Oracle f_A(a+)", "l2_dist": 0.0, "r1": 21.3},
    ]


def table_v_fol_parser() -> list[dict[str, float | str]]:
    """Table 5 — FOL parser analysis on 500 held-out Clotho captions."""
    return [
        {"variant": "Flan-T5-XXL (uncond.)", "em": 54.2, "pa": 61.3, "fb": 9.8, "r1": 18.1},
        {"variant": "+ Vaudio conditioning", "em": 63.8, "pa": 70.1, "fb": 7.4, "r1": 19.3},
        {"variant": "+ domain FT (ours)", "em": 71.4, "pa": 77.6, "fb": 5.1, "r1": 20.4},
    ]


def table_vi_stage3_captioning() -> list[dict[str, float | str]]:
    """Table 6 — Stage 3 captioning sensitivity on Clotho."""
    return [
        {"caption_source": "No re-ranking (S1+S2 only)", "r1": 19.6, "r5": 45.1, "map10": 31.1, "delta_r1": 0.0},
        {"caption_source": "Pengi", "r1": 20.4, "r5": 46.3, "map10": 32.5, "delta_r1": 0.8},
        {"caption_source": "Second captioner", "r1": 20.1, "r5": 45.9, "map10": 32.1, "delta_r1": 0.5},
        {"caption_source": "Oracle (ground-truth)", "r1": 21.3, "r5": 47.8, "map10": 33.9, "delta_r1": 1.7},
    ]


def table_vii_latency() -> list[dict[str, float | str | None]]:
    """Table 7 — Query-time latency on Clotho."""
    return [
        {"system": "CLAP backbone", "index_min": None, "query_ms": 10.0, "r1": 16.75},
        {"system": "FORTE offline", "index_min": 10.0, "query_ms": 12.0, "r1": 20.4},
        {"system": "FORTE online (B=3,D=2)", "index_min": None, "query_ms": 13.0, "r1": 19.8},
        {"system": "FORTE online (B=5,D=4)", "index_min": None, "query_ms": 13.0, "r1": 20.4},
    ]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_i_main": table_i_main_results(),
        "table_ii_alignment_loss": table_ii_alignment_loss(),
        "table_iii_stage_ablation": table_iii_stage_ablation(),
        "table_iv_anchor_bank": table_iv_anchor_bank(),
        "table_v_fol_parser": table_v_fol_parser(),
        "table_vi_stage3_captioning": table_vi_stage3_captioning(),
        "table_vii_latency": table_vii_latency(),
    }


def evaluation_demo(*, seed: int = 0) -> dict[str, Any]:
    return {
        "framework": framework_card(),
        "prompts": prompt_bundle(),
        "smoke": evaluation_smoke(seed=seed),
        "fol": run_fol_refinement_smoke(seed=seed),
        "alignment": run_alignment_smoke(seed=seed + 1),
        "rerank": run_rerank_smoke(seed=seed + 2),
        "pipeline": run_pipeline_smoke(seed=seed + 3),
        "limitations": LIMITATIONS,
    }


def pipeline_demo(*, seed: int = 0) -> dict[str, Any]:
    return evaluation_demo(seed=seed)
