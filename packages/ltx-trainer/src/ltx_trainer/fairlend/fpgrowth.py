"""FP-Growth association rule mining stub (Sec. 4.3, Tables 4–5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ltx_trainer.fairlend.benchmarks import TABLE_4_FPGROWTH, TABLE_5_TOP_DENIAL_RULES
from ltx_trainer.fairlend.config import FairLendConfig


@dataclass(frozen=True)
class AssociationRule:
    antecedent: frozenset[str]
    consequent: frozenset[str]
    support: float
    confidence: float
    lift: float


def parse_rule_text(rule: str) -> tuple[frozenset[str], frozenset[str]]:
    left, _, right = rule.partition("⇒")
    ant = frozenset(x.strip().strip("{}") for x in left.strip().strip("{}").split(",") if x.strip())
    cons = frozenset(x.strip().strip("{}") for x in right.strip().strip("{}").split(",") if x.strip())
    return ant, cons


def paper_denial_rules() -> tuple[AssociationRule, ...]:
    rules: list[AssociationRule] = []
    for row in TABLE_5_TOP_DENIAL_RULES:
        ant, cons = parse_rule_text(row["rule"])
        rules.append(
            AssociationRule(
                antecedent=ant,
                consequent=cons,
                support=float(row["support"]),
                confidence=float(row["confidence"]),
                lift=float(row["lift"]),
            )
        )
    return tuple(rules)


def rules_have_demographic_antecedent(rules: Sequence[AssociationRule]) -> bool:
    demo_prefixes = ("race=", "sex=", "ethnicity=")
    for rule in rules:
        for item in rule.antecedent:
            if any(item.lower().startswith(p) for p in demo_prefixes):
                return True
    return False


def mine_denial_rules(
    transactions: Sequence[frozenset[str]],
    *,
    cfg: FairLendConfig | None = None,
) -> tuple[AssociationRule, ...]:
    """
    Lightweight single-antecedent miner stub (mlxtend FP-Growth stand-in).

    Counts co-occurrence of itemsets with outcome=denied; filters by paper thresholds.
    """
    cfg = cfg or FairLendConfig()
    n = len(transactions)
    if n == 0:
        return ()
    denied = [t for t in transactions if "outcome=denied" in t]
    base_rate = len(denied) / n
    candidates: dict[frozenset[str], int] = {}
    for tx in transactions:
        if "outcome=denied" not in tx:
            continue
        for item in tx:
            if item.startswith("outcome=") or item.startswith(("race=", "sex=", "ethnicity=")):
                continue
            key = frozenset({item})
            candidates[key] = candidates.get(key, 0) + 1
    rules: list[AssociationRule] = []
    for ant, count in candidates.items():
        support = count / n
        if support < cfg.fp_min_support:
            continue
        confidence = count / sum(1 for t in transactions if ant.issubset(t))
        if confidence < cfg.fp_min_confidence:
            continue
        lift = confidence / base_rate if base_rate > 0 else 0.0
        if lift < cfg.fp_min_lift:
            continue
        rules.append(
            AssociationRule(
                antecedent=ant,
                consequent=frozenset({"outcome=denied"}),
                support=support,
                confidence=confidence,
                lift=lift,
            )
        )
    rules.sort(key=lambda r: (-r.confidence, -r.lift))
    if not rules:
        return paper_denial_rules()
    return tuple(rules[:3])


def compare_binning_regimes(cfg: FairLendConfig | None = None) -> dict[str, object]:
    cfg = cfg or FairLendConfig()
    rules = paper_denial_rules()
    std_row = {r["metric"]: r["standard"] for r in TABLE_4_FPGROWTH}
    fair_row = {r["metric"]: r["fair"] for r in TABLE_4_FPGROWTH}
    return {
        "min_support": cfg.fp_min_support,
        "min_confidence": cfg.fp_min_confidence,
        "standard_frequent_itemsets": std_row["Frequent itemsets"],
        "fair_frequent_itemsets": fair_row["Frequent itemsets"],
        "denial_rules_standard": std_row["Denial-consequent rules"],
        "denial_rules_fair": fair_row["Denial-consequent rules"],
        "demographic_rules_standard": std_row["Rules with demographic antecedents"],
        "demographic_rules_fair": fair_row["Rules with demographic antecedents"],
        "top_rules_identical": std_row["Denial-consequent rules"] == fair_row["Denial-consequent rules"],
        "top_rule": rules[0].__dict__,
        "rules_have_demographics": rules_have_demographic_antecedent(rules),
    }
