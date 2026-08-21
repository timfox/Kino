"""Disparate Impact Ratio audit within clusters (Sec. 2.4, Tables 8–9)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ltx_trainer.fairlend.benchmarks import TABLE_8_DIR_FINDINGS, TABLE_9_DIR_SUMMARY
from ltx_trainer.fairlend.config import FairLendConfig


class DirFlag(str, Enum):
    OK = "ok"
    DISPARATE_IMPACT = "DISPARATE_IMPACT"


@dataclass(frozen=True)
class DirAuditRow:
    cluster: int
    attribute: str
    group: str
    reference: str
    group_denial_rate: float
    reference_denial_rate: float
    dir_value: float
    group_size: int
    flag: DirFlag


def disparate_impact_ratio(group_denial: float, ref_denial: float) -> float:
    """Equation 3: acceptance rate ratio (EEOC four-fifths rule)."""
    group_accept = 1.0 - group_denial
    ref_accept = 1.0 - ref_denial
    if ref_accept <= 0:
        return 0.0
    return group_accept / ref_accept


def audit_row(
    cluster: int,
    attribute: str,
    group: str,
    reference: str,
    group_denial: float,
    ref_denial: float,
    group_size: int,
    *,
    threshold: float = 0.80,
) -> DirAuditRow:
    dir_val = disparate_impact_ratio(group_denial, ref_denial)
    flag = DirFlag.DISPARATE_IMPACT if dir_val < threshold else DirFlag.OK
    return DirAuditRow(
        cluster=cluster,
        attribute=attribute,
        group=group,
        reference=reference,
        group_denial_rate=group_denial,
        reference_denial_rate=ref_denial,
        dir_value=dir_val,
        group_size=group_size,
        flag=flag,
    )


def compute_dir_from_assignments(*args, **kwargs):
    """Live DIR audit from applicant records (delegates to cluster_assign)."""
    from ltx_trainer.fairlend.cluster_assign import compute_cluster_dir_audit

    return compute_cluster_dir_audit(*args, **kwargs)


def paper_dir_findings(cfg: FairLendConfig | None = None) -> tuple[DirAuditRow, ...]:
    cfg = cfg or FairLendConfig()
    rows: list[DirAuditRow] = []
    for r in TABLE_8_DIR_FINDINGS:
        rows.append(
            audit_row(
                cluster=int(r["cluster"]),
                attribute="Race",
                group=str(r["group"]),
                reference="White",
                group_denial=float(r["grp_denial"]),
                ref_denial=float(r["ref_denial"]),
                group_size=int(r["size"]),
                threshold=cfg.dir_threshold,
            )
        )
    return tuple(rows)


def dir_audit_summary(cfg: FairLendConfig | None = None) -> dict[str, object]:
    cfg = cfg or FairLendConfig()
    findings = paper_dir_findings(cfg)
    flagged = [f for f in findings if f.flag == DirFlag.DISPARATE_IMPACT]
    race_summary = next(r for r in TABLE_9_DIR_SUMMARY if r["attribute"] == "Race")
    return {
        "threshold": cfg.dir_threshold,
        "min_group_size": cfg.dir_min_group_size,
        "pairs_audited": cfg.dir_audited_pairs,
        "flagged_count": len(flagged),
        "paper_flagged_count": 10,
        "race_mean_dir": race_summary["mean_dir"],
        "worst_dir": min(f.dir_value for f in findings),
        "cluster3_black_dir": next(
            f.dir_value for f in findings if f.cluster == 3 and "Black" in f.group
        ),
        "findings": [f.__dict__ for f in findings],
        "summary_by_attribute": TABLE_9_DIR_SUMMARY,
    }
