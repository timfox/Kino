"""Deployment-phase update stub (Sec. 5.5 Phase II)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ltx_trainer.tahoe.atomic_diff import make_column_choice_diff
from ltx_trainer.tahoe.learning import AtomicDiff


class DeploymentOutcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ALL_REJECTED = "all_rejected"


@dataclass(frozen=True)
class CandidateSQL:
    sql: str
    outcome: DeploymentOutcome


@dataclass
class DeploymentFeedbackBatch:
    question: str
    database_id: str | None
    candidates: tuple[CandidateSQL, ...]


def classify_deployment_batch(batch: DeploymentFeedbackBatch) -> dict[str, object]:
    """Human-in-the-loop semantic learning signal (Sec. 5.5)."""
    accepted = [c for c in batch.candidates if c.outcome == DeploymentOutcome.ACCEPTED]
    rejected = [c for c in batch.candidates if c.outcome == DeploymentOutcome.REJECTED]
    if not accepted and any(c.outcome == DeploymentOutcome.ALL_REJECTED for c in batch.candidates):
        return {"action": "manual_review", "semantic_diffs": [], "pseudo_ground_truth": None}
    if not accepted:
        return {"action": "manual_review", "semantic_diffs": [], "pseudo_ground_truth": None}
    pseudo = accepted[0].sql
    diffs: list[AtomicDiff] = []
    if batch.database_id == "GA4" and rejected:
        diffs.append(make_column_choice_diff())
    return {
        "action": "semantic_learn" if diffs else "log_only",
        "semantic_diffs": diffs,
        "pseudo_ground_truth": pseudo,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
    }


def run_deployment_ga4_demo() -> dict[str, object]:
    batch = DeploymentFeedbackBatch(
        question="List unique visitors from last month",
        database_id="GA4",
        candidates=(
            CandidateSQL(
                "SELECT DISTINCT USER_ID FROM GA4.EVENTS",
                DeploymentOutcome.REJECTED,
            ),
            CandidateSQL(
                'SELECT DISTINCT "USER_PSEUDO_ID" FROM "GA4"."EVENTS"',
                DeploymentOutcome.ACCEPTED,
            ),
        ),
    )
    return classify_deployment_batch(batch)
