"""Extended lift and low-support rule analysis (Sec. 5.2, 6.2, Sec. 7)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.fairlend.fpgrowth import AssociationRule, mine_denial_rules
from ltx_trainer.fairlend.transactions import ApplicationRecord, build_transaction, synthetic_chicago_applications


@dataclass(frozen=True)
class ExtendedLiftRow:
    group: str
    support: float
    confidence: float
    lift: float
    rule_item: str


def _denial_rate(records: tuple[ApplicationRecord, ...]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r.denied) / len(records)


def extended_lift_for_item(
    records: tuple[ApplicationRecord, ...],
    item: str,
    *,
    group_key: str = "race",
    group_value: str = "Black or African American",
) -> ExtendedLiftRow | None:
    """Lift of item → denied within a demographic subgroup."""
    if group_key == "race":
        sub = tuple(r for r in records if r.race == group_value)
    else:
        return None
    if not sub:
        return None
    txs = [build_transaction(r) for r in sub]
    n = len(txs)
    base = _denial_rate(sub)
    with_item = [t for t in txs if item in t]
    if not with_item:
        return ExtendedLiftRow(group=group_value, support=0.0, confidence=0.0, lift=0.0, rule_item=item)
    support = len(with_item) / n
    denied_with = sum(1 for t in with_item if "outcome=denied" in t)
    confidence = denied_with / len(with_item)
    lift = confidence / base if base > 0 else 0.0
    return ExtendedLiftRow(
        group=group_value,
        support=support,
        confidence=confidence,
        lift=lift,
        rule_item=item,
    )


def low_support_demographic_probe(
    records: tuple[ApplicationRecord, ...],
    *,
    support_threshold: float = 0.05,
) -> dict[str, object]:
    """
    Sec. 6.2: at 10% support no race antecedents; at 3–5% minority rules may appear.
    """
    txs = [build_transaction(r) for r in records]
    cfg_rules_high = mine_denial_rules(txs)
    from ltx_trainer.fairlend.config import FairLendConfig

    cfg_low = FairLendConfig(fp_min_support=support_threshold)
    rules_low = mine_denial_rules(txs, cfg=cfg_low)
    dti_high = extended_lift_for_item(records, "dti_bin=DTI_High")
    dti_black = extended_lift_for_item(records, "dti_bin=DTI_High", group_value="Black or African American")
    dti_white = extended_lift_for_item(records, "dti_bin=DTI_High", group_value="White")
    return {
        "support_threshold_high": 0.10,
        "support_threshold_low": support_threshold,
        "rules_at_10pct": len(cfg_rules_high),
        "rules_at_low_support": len(rules_low),
        "top_rule_high_support": cfg_rules_high[0].__dict__ if cfg_rules_high else None,
        "extended_lift_black_dti": dti_black.__dict__ if dti_black else None,
        "extended_lift_white_dti": dti_white.__dict__ if dti_white else None,
        "extended_lift_overall_dti": dti_high.__dict__ if dti_high else None,
        "black_denied_share": round(
            sum(1 for r in records if r.race == "Black or African American" and r.denied)
            / max(1, sum(1 for r in records if r.race == "Black or African American")),
            4,
        ),
    }


def extended_lift_demo() -> dict[str, object]:
    records = synthetic_chicago_applications(n=1500)
    return low_support_demographic_probe(records, support_threshold=0.05)
