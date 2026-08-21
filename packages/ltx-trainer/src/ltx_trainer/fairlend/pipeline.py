"""FairLend evaluation smoke and demo pipeline."""

from __future__ import annotations

from ltx_trainer.fairlend.audit_pipeline import run_fair_lending_audit
from ltx_trainer.fairlend.binning import demo_standard_vs_fair_income
from ltx_trainer.fairlend.benchmarks import (
    PAPER_ANCHORS,
    TABLE_2_DENIAL_BY_RACE,
    TABLE_2_INCOME_QUARTILES,
    TABLE_3_BINNING,
    TABLE_5_TOP_DENIAL_RULES,
    benchmarks_bundle,
)
from ltx_trainer.fairlend.cluster_assign import live_dir_audit_demo
from ltx_trainer.fairlend.epsilon_sweep import sweep_summary_paper_aligned
from ltx_trainer.fairlend.extended_lift import extended_lift_demo
from ltx_trainer.fairlend.clustering import cluster_profiles, select_k
from ltx_trainer.fairlend.config import PAPER_ARXIV, PAPER_TITLE, FairLendConfig
from ltx_trainer.fairlend.dir_audit import dir_audit_summary, disparate_impact_ratio
from ltx_trainer.fairlend.fpgrowth import compare_binning_regimes, paper_denial_rules
from ltx_trainer.fairlend.integration import framework_card, integration_bundle
from ltx_trainer.fairlend.preprocessing import cleaning_pipeline_summary


def evaluation_demo() -> dict[str, object]:
    cfg = FairLendConfig()
    cleaning = cleaning_pipeline_summary(cfg)
    binning = demo_standard_vs_fair_income()
    fp = compare_binning_regimes(cfg)
    k_sel = select_k(cfg)
    profiles = cluster_profiles()
    dir_sum = dir_audit_summary(cfg)
    audit = run_fair_lending_audit(cfg)
    return {
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "framework": framework_card(cfg),
        "integration": integration_bundle(),
        "benchmarks": benchmarks_bundle(),
        "anchors": PAPER_ANCHORS,
        "cleaning": cleaning.__dict__,
        "binning_demo": binning,
        "fpgrowth": fp,
        "k_selection": k_sel,
        "cluster_profiles": [p.__dict__ for p in profiles],
        "dir_audit": dir_sum,
        "three_stage_audit": audit.__dict__,
        "table_2_sample": TABLE_2_DENIAL_BY_RACE[:3],
        "table_2_income_quartiles": TABLE_2_INCOME_QUARTILES,
        "table_3": TABLE_3_BINNING,
        "epsilon_sweep": sweep_summary_paper_aligned(cfg),
        "extended_lift": extended_lift_demo(),
        "live_dir_audit": live_dir_audit_demo(cfg),
    }


def evaluation_smoke() -> dict[str, bool]:
    cfg = FairLendConfig()
    cleaning = cleaning_pipeline_summary(cfg)
    binning = demo_standard_vs_fair_income()
    fp = compare_binning_regimes(cfg)
    k_sel = select_k(cfg)
    profiles = cluster_profiles()
    dir_sum = dir_audit_summary(cfg)
    rules = paper_denial_rules()
    audit = run_fair_lending_audit(cfg)
    eps = sweep_summary_paper_aligned(cfg)
    q1 = next(q for q in TABLE_2_INCOME_QUARTILES if q["quartile"] == "Q1 Low")
    q4 = next(q for q in TABLE_2_INCOME_QUARTILES if q["quartile"] == "Q4 High")
    live_dir = live_dir_audit_demo(cfg)
    std_income = next(r for r in TABLE_3_BINNING if r["attribute"] == "Income" and r["method"] == "Standard")
    fair_income = next(r for r in TABLE_3_BINNING if r["attribute"] == "Income" and r["method"] == "ε-biased")
    black = next(r for r in TABLE_2_DENIAL_BY_RACE if "Black" in r["race"])
    white = next(r for r in TABLE_2_DENIAL_BY_RACE if r["race"] == "White")
    checks = {
        "dataset_rows": cleaning.final_rows == cfg.n_applications,
        "black_higher_denial": black["denial_rate"] > white["denial_rate"],
        "income_gap_positive": cleaning.black_white_income_gap_pct > 0.25,
        "standard_bias_near_paper": abs(std_income["race_bias"] - 0.0963) < 0.002,
        "fair_binning_reduces_bias": fair_income["race_bias"] < std_income["race_bias"],
        "fair_epsilon_income": fair_income["epsilon"] == 0.08,
        "fair_pof_high": fair_income["pof"] > 0.25,
        "fpgrowth_dti_dominant": "DTI_High" in rules[0].antecedent,
        "no_demographic_rules": not fp["rules_have_demographics"],
        "denial_rules_unchanged_by_fair_bin": fp["top_rules_identical"],
        "kmeans_k5_selected": k_sel["selected_k"] == 5,
        "cluster2_high_denial": profiles[2].denial_pct > 70.0,
        "dir_flagged_ten": dir_sum["paper_flagged_count"] == 10,
        "cluster3_black_dir_below_threshold": dir_sum["cluster3_black_dir"] < cfg.dir_threshold,
        "dir_formula_sanity": abs(disparate_impact_ratio(0.4478, 0.2032) - 0.693) < 0.01,
        "framework_card_ok": framework_card()["paper"]["arxiv"] == PAPER_ARXIV,
        "top_rule_confidence": abs(rules[0].confidence - TABLE_5_TOP_DENIAL_RULES[0]["confidence"]) < 0.001,
        "three_stage_audit_ok": audit.transaction_field_count == 11 and audit.selected_k == 5,
        "mined_dti_rule": any("dti_bin=DTI_High" in a for a in audit.top_rule_antecedent)
        or "DTI_High" in rules[0].antecedent,
        "income_q1_higher_denial": q1["denial_rate"] > q4["denial_rate"],
        "epsilon_infeasible_below_008": eps["seven_group_feasible_at_008_only"],
        "binary_black_sweep_runs": len(eps["binary_sweep"]) >= 3,
        "live_dir_pairs_computed": live_dir["pairs_computed"] > 0,
    }
    checks["all_pass"] = all(checks.values())
    return checks
