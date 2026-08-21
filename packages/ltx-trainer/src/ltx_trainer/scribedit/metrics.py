"""VIBE-style score helpers and cross-task transfer tables (Sec. 3, 5)."""

from __future__ import annotations

import math
from typing import Iterable

from ltx_trainer.scribedit.config import EditTask, ScribEditConfig


def geometric_mean_score(scores: Iterable[float]) -> float:
    """VIBE Deictic-Level aggregation (geometric mean over sub-criteria)."""
    vals = [max(s, 1e-6) for s in scores]
    if not vals:
        return 0.0
    log_sum = sum(math.log(v) for v in vals)
    return math.exp(log_sum / len(vals))


def single_task_average(task_scores: dict[EditTask, float]) -> float:
    """1-Task column: average over AD, RM, RP, TR (Table 4)."""
    if not task_scores:
        return 0.0
    return sum(task_scores.values()) / len(task_scores)


def cross_task_transfer_matrix(
    train_task: EditTask,
    eval_scores: dict[EditTask, float],
) -> dict[str, float]:
    """Row for Table 1: train on one task, eval on all."""
    return {"train": train_task, **{f"eval_{k}": eval_scores[k] for k in eval_scores}}


def instruction_vs_domain_gap(
    *,
    instruction_adherence_synthetic: float,
    instruction_adherence_real: float,
    visual_coherence_synthetic: float,
    visual_coherence_real: float,
) -> dict[str, float]:
    """Study 2 gaps: IA vs VC synthetic→real (Table 2)."""
    return {
        "instruction_adherence_drop": instruction_adherence_real - instruction_adherence_synthetic,
        "visual_coherence_drop": visual_coherence_real - visual_coherence_synthetic,
    }


def distractor_ablation_delta(
    without_distractor: dict[str, float],
    with_distractor: dict[str, float],
) -> dict[str, float]:
    """Table 3: score deltas for 2-Task / 3-Task with visual distractors."""
    keys = set(without_distractor) & set(with_distractor)
    return {k: with_distractor[k] - without_distractor[k] for k in keys}


def baseline_cross_task_score(
    train_task: EditTask,
    eval_task: EditTask,
    *,
    cfg: ScribEditConfig | None = None,
) -> float | None:
    """Lookup published single-task-only fine-tune scores (Table 1 subset)."""
    cfg = cfg or ScribEditConfig()
    # Published diagonal / notable off-diagonal from paper Table 1
    table: dict[tuple[str, str], float] = {
        ("AD", "AD"): 86.50,
        ("AD", "RM"): 37.75,
        ("AD", "TR"): 9.43,
        ("RM", "RM"): 97.18,
        ("RM", "TR"): 16.99,
        ("TR", "TR"): 84.01,
    }
    return table.get((train_task, eval_task))
