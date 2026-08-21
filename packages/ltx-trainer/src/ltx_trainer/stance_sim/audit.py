"""Four-stage stance simulation audit pipeline (arXiv:2606.06443)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.stance_sim.metrics import (
    average_directional_shift,
    classification_metrics,
    directional_stance_shift,
)
from ltx_trainer.stance_sim.mock import ToyStanceSimulator, run_revision_audit, sample_instances
from ltx_trainer.stance_sim.revision import ConversationInstance, RevisionStrategy, simulate_revised_stance


def run_stage1_validation(
    instances: list[ConversationInstance],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Stage 1 — inferred (masked) vs observed (full) stance agreement."""
    sim = ToyStanceSimulator(seed=seed)
    observed = [inst.observed_stance for inst in instances]
    inferred = [sim.infer(inst, include_target_message=False) for inst in instances]
    observed_sim = [sim.infer(inst, include_target_message=True) for inst in instances]
    metrics = classification_metrics(observed, inferred)
    return {
        "n": len(instances),
        "inferred_vs_observed": metrics,
        "simulated_observed_vs_recorded": classification_metrics(observed, observed_sim),
    }


def run_four_stage_audit(
    instances: list[ConversationInstance] | None = None,
    *,
    seed: int = 0,
    strategies: tuple[RevisionStrategy, ...] = (
        RevisionStrategy.PARAPHRASE,
        RevisionStrategy.EXPLAIN,
        RevisionStrategy.ADD,
        RevisionStrategy.MEME,
    ),
) -> dict[str, Any]:
    """Stages 1–4: validate, revise, re-simulate, aggregate shift metrics."""
    instances = instances or sample_instances(seed=seed, n=32)
    sim = ToyStanceSimulator(seed=seed)
    stage1 = run_stage1_validation(instances, seed=seed)

    inferred = [sim.infer(inst, include_target_message=False) for inst in instances]
    revisions: dict[str, Any] = {}
    all_pairs: list[tuple[str, str]] = []

    for i, strategy in enumerate(strategies):
        audit = run_revision_audit(instances, strategy, seed=seed + i + 1)
        revisions[strategy.value] = audit
        for j, inst in enumerate(instances):
            y_inf = sim.infer(inst, include_target_message=False)
            y_rev = simulate_revised_stance(y_inf, strategy, seed=seed + i * 100 + j)
            all_pairs.append((y_inf, y_rev))

    deltas = [directional_stance_shift(a, b) for a, b in all_pairs]
    return {
        "stage1": stage1,
        "stage2_4_revisions": revisions,
        "mean_delta_all_strategies": round(float(sum(deltas) / len(deltas)), 4) if deltas else 0.0,
        "average_directional_shift": round(average_directional_shift(all_pairs), 4),
        "instances": len(instances),
    }


def figure3_transition_rates() -> dict[str, dict[str, float]]:
    """Fig. 3 — stance transition rates by revision (GPT-5.2, paper headline)."""
    return {
        "paraphrase": {
            "negative_to_neutral_or_positive_pct": 2.1,
            "neutral_to_positive_pct": 1.4,
            "neutral_to_negative_pct": 1.8,
        },
        "explain": {
            "negative_to_neutral_or_positive_pct": 5.6,
            "neutral_to_positive_pct": 3.2,
            "neutral_to_negative_pct": 2.0,
        },
        "add": {
            "negative_to_neutral_or_positive_pct": 4.7,
            "neutral_to_positive_pct": 6.1,
            "neutral_to_negative_pct": 3.4,
        },
        "meme": {
            "negative_to_neutral_or_positive_pct": 8.2,
            "neutral_to_positive_pct": 17.6,
            "neutral_to_negative_pct": 6.9,
        },
    }
