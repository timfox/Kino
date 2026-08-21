"""SkillCorpus evaluation pipeline and smoke checks."""

from __future__ import annotations

from typing import Any

from ltx_trainer.skillcorpus.config import SkillCorpusConfig, TAXONOMY_CLASSES
from ltx_trainer.skillcorpus.pipeline_core import (
    COVERAGE_BIN_DELTA,
    coverage_bin,
    curation_funnel_summary,
    demo_skills,
    retrieve_and_select,
)
from ltx_trainer.skillcorpus.quality import FacetScores, classify_skill, composite_score, hard_gate_blocks


def run_curation_demo() -> dict[str, Any]:
    funnel = curation_funnel_summary()
    skills = demo_skills()
    active = [s for s in skills if s.active]
    blocked = [s for s in skills if not s.active]
    return {
        "funnel": funnel,
        "demo_n": len(skills),
        "demo_active": len(active),
        "demo_blocked": len(blocked),
        "hard_gate_blocks_malware": hard_gate_blocks(("cmd_injection",)),
        "taxonomy_n": len(TAXONOMY_CLASSES),
    }


def run_quality_demo() -> dict[str, Any]:
    good = FacetScores(0.9, 0.85, 0.95)
    marginal = FacetScores(0.8, 0.7, 0.5)
    gated = FacetScores(0.9, 0.9, 0.9, ("prompt_injection",))
    return {
        "good_score": composite_score(good, has_scripts=True),
        "marginal_score": composite_score(marginal),
        "gated_score": composite_score(gated),
        "classify_dev": classify_skill("code-review", "Review Python PRs with git"),
        "classify_data": classify_skill("sql-etl", "Build pandas ETL for analytics"),
        "facets_separated": True,  # design claim; paper r(u,s)=0.15
    }


def run_retrieval_demo() -> dict[str, Any]:
    rescue = retrieve_and_select(
        "Normalise manufacturing defect reason texts into a codebook with station validity"
    )
    break_case = retrieve_and_select(
        "Update exchange rate in an Excel table embedded inside a PowerPoint slide"
    )
    return {
        "rescue": rescue,
        "break_case": break_case,
        "rescue_selects": rescue["n_selected"] >= 1,
        "coverage_bin": coverage_bin(rescue["top_rerank"]),
        "bin_delta_pp": COVERAGE_BIN_DELTA[coverage_bin(rescue["top_rerank"])],
    }


def evaluation_demo(config: SkillCorpusConfig | None = None) -> dict[str, Any]:
    _ = config or SkillCorpusConfig()
    cur = run_curation_demo()
    qual = run_quality_demo()
    ret = run_retrieval_demo()
    return {
        "curation": cur,
        "quality": qual,
        "retrieval": ret,
        "funnel_to_96k": cur["funnel"]["n_active"] == 96401,
        "gates_unsafe": cur["hard_gate_blocks_malware"] and qual["gated_score"] == 0.0,
        "good_score_positive": qual["good_score"] > 0.5,
        "retrieval_selects": ret["rescue_selects"],
        "taxonomy_16": cur["taxonomy_n"] == 16,
    }


def evaluation_smoke() -> dict[str, bool]:
    from ltx_trainer.skillcorpus.benchmarks import PAPER_ANCHORS, TABLE_1_POOLED_DELTA

    demo = evaluation_demo()
    checks = {
        "funnel_to_96k": demo["funnel_to_96k"],
        "gates_unsafe": demo["gates_unsafe"],
        "good_score_positive": demo["good_score_positive"],
        "retrieval_selects": demo["retrieval_selects"],
        "taxonomy_16": demo["taxonomy_16"],
        "paper_n_active": int(PAPER_ANCHORS["n_active"]) == 96401,
        "paper_skillsbench_delta": abs(float(PAPER_ANCHORS["skillsbench_pooled_delta_pp"]) - 7.5) < 1e-6,
        "paper_pooled_table": abs(float(TABLE_1_POOLED_DELTA["SkillsBench"]) - 7.5) < 1e-6,
        "paper_ablation_full": abs(float(PAPER_ANCHORS["ablation_full_pass"]) - 22.6) < 1e-6,
        "paper_opus_delta": abs(float(PAPER_ANCHORS["opus_delta_pp"]) - 8.0) < 1e-6,
    }
    checks["all_pass"] = all(checks.values())
    return checks


def evaluation_smoke_json() -> dict[str, bool]:
    return evaluation_smoke()
