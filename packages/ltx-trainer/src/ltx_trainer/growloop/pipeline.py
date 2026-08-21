"""GrowLoop framework card and paper tables (arXiv:2605.28882)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.growloop.config import GrowLoopConfig
from ltx_trainer.growloop.csp import csp_field_registry, csp_pool_sources
from ltx_trainer.growloop.dual_loop import DUAL_LOOP_TRIGGERS, dual_loop_smoke
from ltx_trainer.growloop.dual_loop_runner import dual_loop_runner_smoke
from ltx_trainer.growloop.gates import gate_registry, gates_smoke
from ltx_trainer.growloop.heuristic import heuristic_smoke, stage_pipeline
from ltx_trainer.growloop.layout import LIMITATIONS
from ltx_trainer.growloop.metrics import metrics_smoke
from ltx_trainer.growloop.zones import zones_smoke


def framework_card(cfg: GrowLoopConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GrowLoopConfig()
    return {
        "name": "GrowLoop",
        "paper": cfg.paper_arxiv,
        "org": cfg.org,
        "idea": (
            "Self-evolving conversation evaluation seeded by human annotations. "
            "Heuristic Learning externalizes tacit human-likeness criteria into "
            "Rubric_safety + Rubric_quality; CSP-driven case generation and "
            "dual-loop co-evolution keep rubrics and cases adapting as models advance."
        ),
        "zones": {
            "consensus": "Require human–AI agreement",
            "divergence": "Require plausibility only",
            "inter_annotator_agreement": cfg.inter_annotator_agreement,
        },
        "rubric": {
            "dimensions": cfg.quality_dimensions,
            "cognitive_categories": cfg.cognitive_categories,
            "seed_cases": cfg.seed_cases,
            "merged_agreement_gemini": cfg.merged_agreement_gemini,
        },
        "cases": {
            "count": cfg.case_count,
            "csp_fields": cfg.csp_fields,
            **csp_pool_sources(cfg),
        },
        "stages": stage_pipeline(cfg),
        "judge_llm": cfg.judge_llm,
        "limitations": list(LIMITATIONS),
    }


def table_vii_cross_model() -> list[dict[str, Any]]:
    """Table 7 — judge agreement under same rubric."""
    return [
        {"judge": "Gemini 3.1 Pro Preview", "step1_pct": 95.3, "step2_pct": 85.7, "merge_pct": 86.0},
        {"judge": "Claude Opus 4.6", "step1_pct": 96.2, "step2_pct": 81.8, "merge_pct": 83.6},
    ]


def table_viii_baselines() -> list[dict[str, Any]]:
    """Table 8 — pairwise judge comparison (132 cases)."""
    return [
        {"category": "No rubric", "method": "Zero-shot", "tie_aware": 0.25, "pair_acc": 0.42, "spearman": -0.21},
        {"category": "No rubric", "method": "ICL (k=3)", "tie_aware": 0.37, "pair_acc": 0.57, "spearman": 0.09},
        {"category": "Manual rubric", "method": "Arena-Hard", "tie_aware": 0.27, "pair_acc": 0.43, "spearman": -0.17},
        {"category": "Training-free", "method": "ICAI", "tie_aware": 0.58, "pair_acc": 0.85, "spearman": 0.75},
        {"category": "Reward Model", "method": "RM-R1", "tie_aware": 0.15, "pair_acc": 0.25, "spearman": -0.50},
        {"category": "Ours", "method": "GrowLoop", "tie_aware": 0.78, "pair_acc": 0.87, "spearman": 0.78},
    ]


def table_ix_capabilities() -> list[dict[str, str]]:
    """Table 9 — qualitative capability matrix."""
    return [
        {"capability": "Interpretable", "zero_shot": "△", "arena_hard": "✓", "icai": "✓", "rm_r1": "×", "growloop": "✓"},
        {"capability": "Dimensional", "zero_shot": "×", "arena_hard": "×", "icai": "✓", "rm_r1": "×", "growloop": "✓"},
        {"capability": "Editable", "zero_shot": "×", "arena_hard": "✓", "icai": "✓", "rm_r1": "×", "growloop": "✓"},
        {"capability": "Evolvable", "zero_shot": "×", "arena_hard": "×", "icai": "×", "rm_r1": "×", "growloop": "✓"},
        {"capability": "Tacit-aware", "zero_shot": "×", "arena_hard": "×", "icai": "△", "rm_r1": "×", "growloop": "✓"},
    ]


def table_xi_tier_profile(cfg: GrowLoopConfig | None = None) -> list[dict[str, Any]]:
    """Table 11 — per-tier means and fatal rates."""
    cfg = cfg or GrowLoopConfig()
    tiers = ("Best", "Good", "Medium", "Bad")
    return [
        {"tier": t, "mean_score": m, "fatal_pct": f}
        for t, m, f in zip(tiers, cfg.tier_means, cfg.tier_fatal_pct, strict=True)
    ]


def table_xiv_feedback_ablation() -> list[dict[str, Any]]:
    """Table 14 — case evolution feedback components."""
    return [
        {"config": "Full system", "critic": True, "inter_batch": True, "cross_round": True, "tau": 0.713},
        {"config": "Critic only", "critic": True, "inter_batch": False, "cross_round": False, "tau": 0.48},
        {"config": "Inter-batch only", "critic": False, "inter_batch": True, "cross_round": False, "tau": 0.42},
        {"config": "None", "critic": False, "inter_batch": False, "cross_round": False, "tau": 0.28},
    ]


def headline_results(cfg: GrowLoopConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GrowLoopConfig()
    return {
        "consensus_merge_agreement": f"{cfg.merged_agreement_gemini * 100:.1f}% (Gemini judge, Table 7)",
        "growloop_tie_aware": f"GrowLoop {cfg.growloop_tie_aware_acc}",
        "vs_icai_tie_aware": f"GrowLoop {cfg.growloop_tie_aware_acc} vs ICAI {cfg.icai_tie_aware_acc} (+20 pp)",
        "best_pairwise": f"Tie-aware {cfg.growloop_tie_aware_acc}, Pair-Acc {cfg.growloop_pair_acc}, ρ {cfg.growloop_spearman}",
        "gates_pass": "All five hard gates pass at 500-case scale (Table 5)",
        "key_finding": (
            "Scalar reward models anti-correlate with human-likeness; explicit evolving rubrics "
            "externalize tacit criteria and stay discriminative across model tiers"
        ),
        "only_evolvable": "Only method satisfying interpretable + dimensional + editable + evolvable + tacit-aware",
    }


def pipeline_demo(cfg: GrowLoopConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or GrowLoopConfig()
    return {
        "csp_fields": len(csp_field_registry(cfg)),
        "case_count": cfg.case_count,
        "heuristic": heuristic_smoke(cfg),
        "zones": zones_smoke(cfg),
        "gates": gates_smoke(cfg),
        "metrics": metrics_smoke(cfg, seed=seed),
        "dual_loop": dual_loop_smoke(cfg),
        "dual_loop_runner": dual_loop_runner_smoke(cfg),
        "growloop_tie_aware_anchor": cfg.growloop_tie_aware_acc,
    }


def evaluation_demo(*, seed: int = 42) -> dict[str, Any]:
    return {
        "headline": headline_results(),
        "demo": pipeline_demo(seed=seed),
        "framework": framework_card(),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "csp_fields": csp_field_registry(),
        "dual_loop_triggers": list(DUAL_LOOP_TRIGGERS),
        "hard_gates": gate_registry(),
        "table_vii_cross_model": table_vii_cross_model(),
        "table_viii_baselines": table_viii_baselines(),
        "table_ix_capabilities": table_ix_capabilities(),
        "table_xi_tier_profile": table_xi_tier_profile(),
        "table_xiv_feedback_ablation": table_xiv_feedback_ablation(),
        "stages": stage_pipeline(),
        "headline": headline_results(),
    }
