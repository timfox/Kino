"""End-to-end three-stage FairLend audit pipeline (Sec. 4–5)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.fairlend.binning import demo_standard_vs_fair_income
from ltx_trainer.fairlend.clustering import select_k
from ltx_trainer.fairlend.config import FairLendConfig
from ltx_trainer.fairlend.dir_audit import DirFlag, dir_audit_summary, paper_dir_findings
from ltx_trainer.fairlend.fpgrowth import mine_denial_rules, paper_denial_rules
from ltx_trainer.fairlend.preprocessing import cleaning_pipeline_summary
from ltx_trainer.fairlend.transactions import (
    TRANSACTION_FIELDS,
    synthetic_chicago_applications,
    transactions_from_records,
)


@dataclass(frozen=True)
class FairLendingAuditResult:
    cleaning_rows: int
    standard_income_bias: float
    fair_income_bias: float | None
    mined_denial_rules: int
    top_rule_antecedent: tuple[str, ...]
    selected_k: int
    dir_flagged: int
    dir_worst: float
    transaction_field_count: int


def run_fair_lending_audit(cfg: FairLendConfig | None = None) -> FairLendingAuditResult:
    cfg = cfg or FairLendConfig()
    cleaning = cleaning_pipeline_summary(cfg)
    binning = demo_standard_vs_fair_income()
    records = synthetic_chicago_applications(n=400)
    txs = transactions_from_records(records)
    mined = mine_denial_rules(txs, cfg=cfg)
    paper_rules = paper_denial_rules()
    k_sel = select_k(cfg)
    dir_sum = dir_audit_summary(cfg)
    findings = paper_dir_findings(cfg)
    flagged = [f for f in findings if f.flag == DirFlag.DISPARATE_IMPACT]
    top = paper_rules[0] if paper_rules else (mined[0] if mined else None)
    antecedent = tuple(sorted(top.antecedent)) if top else ()
    return FairLendingAuditResult(
        cleaning_rows=cleaning.final_rows,
        standard_income_bias=float(binning["standard_race_bias"]),
        fair_income_bias=float(binning["fair_race_bias"]) if binning["fair_race_bias"] is not None else None,
        mined_denial_rules=len(mined),
        top_rule_antecedent=antecedent,
        selected_k=int(k_sel["selected_k"]),
        dir_flagged=len(flagged),
        dir_worst=float(dir_sum["worst_dir"]),
        transaction_field_count=len(TRANSACTION_FIELDS),
    )
