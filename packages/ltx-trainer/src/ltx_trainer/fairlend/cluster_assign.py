"""Heuristic cluster assignment + live DIR audit (Sec. 4.4, 5.4)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.fairlend.clustering import zscore_features
from ltx_trainer.fairlend.config import FairLendConfig
from ltx_trainer.fairlend.dir_audit import DirAuditRow, DirFlag, audit_row, disparate_impact_ratio
from ltx_trainer.fairlend.transactions import ApplicationRecord, synthetic_chicago_applications


@dataclass(frozen=True)
class ClusteredApplicant:
    record: ApplicationRecord
    cluster_id: int


def assign_cluster_heuristic(record: ApplicationRecord) -> int:
    """
    Rule-based cluster id aligned with Table 7 centroids (k=5 stub).

    Not full K-Means — maps financial profile to nearest paper cluster profile.
    """
    if record.dti_numeric >= 55:
        return 2
    if record.income_k >= 400:
        return 4
    if record.income_k >= 180 and record.dti_numeric < 40:
        return 1
    if record.loan_amount_k / max(record.income_k, 1.0) < 1.2 and record.dti_numeric < 38:
        return 3
    return 0


def cluster_applicants(records: tuple[ApplicationRecord, ...]) -> tuple[ClusteredApplicant, ...]:
    return tuple(ClusteredApplicant(record=r, cluster_id=assign_cluster_heuristic(r)) for r in records)


def feature_matrix(records: tuple[ApplicationRecord, ...]) -> list[dict[str, float]]:
    return [
        {
            "income": r.income_k,
            "loan": r.loan_amount_k,
            "dti": r.dti_numeric,
            "cltv": min(100.0, r.loan_amount_k / max(r.income_k, 1.0) * 30.0),
        }
        for r in records
    ]


def compute_cluster_dir_audit(
    records: tuple[ApplicationRecord, ...],
    *,
    reference_race: str = "White",
    protected_races: tuple[str, ...] = ("Black or African American",),
    min_group_size: int = 30,
    threshold: float = 0.80,
) -> list[DirAuditRow]:
    """DIR within each cluster for protected vs reference race."""
    clustered = cluster_applicants(records)
    by_cluster: dict[int, list[ClusteredApplicant]] = {}
    for row in clustered:
        by_cluster.setdefault(row.cluster_id, []).append(row)

    findings: list[DirAuditRow] = []
    for cluster_id, rows in sorted(by_cluster.items()):
        ref_rows = [r for r in rows if r.record.race == reference_race]
        if len(ref_rows) < min_group_size:
            continue
        ref_denial = sum(1 for r in ref_rows if r.record.denied) / len(ref_rows)
        for race in protected_races:
            grp_rows = [r for r in rows if r.record.race == race]
            if len(grp_rows) < min_group_size:
                continue
            grp_denial = sum(1 for r in grp_rows if r.record.denied) / len(grp_rows)
            findings.append(
                audit_row(
                    cluster=cluster_id,
                    attribute="Race",
                    group=race,
                    reference=reference_race,
                    group_denial=grp_denial,
                    ref_denial=ref_denial,
                    group_size=len(grp_rows),
                    threshold=threshold,
                )
            )
    return findings


def live_dir_audit_demo(cfg: FairLendConfig | None = None) -> dict[str, object]:
    cfg = cfg or FairLendConfig()
    records = synthetic_chicago_applications(n=2000)
    zscore_features(feature_matrix(records))
    findings = compute_cluster_dir_audit(
        records,
        min_group_size=cfg.dir_min_group_size,
        threshold=cfg.dir_threshold,
    )
    flagged = [f for f in findings if f.flag == DirFlag.DISPARATE_IMPACT]
    return {
        "n_applicants": len(records),
        "pairs_computed": len(findings),
        "flagged": len(flagged),
        "sample_findings": [f.__dict__ for f in findings[:5]],
        "worst_dir": min((f.dir_value for f in findings), default=1.0),
        "cluster3_black": next(
            (f.__dict__ for f in findings if f.cluster == 3 and "Black" in f.group),
            None,
        ),
    }
