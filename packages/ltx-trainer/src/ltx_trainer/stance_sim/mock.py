"""Toy stance simulator and evaluation smoke (arXiv:2606.06443)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.stance_sim.metrics import (
    average_directional_shift,
    classification_metrics,
    depolarization_rate,
    directional_stance_shift,
    supportive_transition_rate,
)
from ltx_trainer.stance_sim.revision import (
    ConversationInstance,
    RevisionStrategy,
    revise_message,
    revise_tone,
    simulate_revised_stance,
)


class ToyStanceSimulator:
    """Deterministic stance model fθ(ui, ti | Ci, ni) without LLM calls."""

    def __init__(self, *, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def infer(
        self,
        instance: ConversationInstance,
        *,
        include_target_message: bool = False,
    ) -> str:
        text = (
            instance.context
            + instance.last_other_message
            + (instance.last_target_message if include_target_message else "")
        ).lower()
        target = instance.stance_target.lower()
        neg_cues = ("concern", "privacy", "censorship", "worried", "against", "crush")
        pos_cues = ("impressive", "good", "potential", "benchmark", "open source")
        score = 0.0
        if target in text:
            score += 0.2
        score += 0.15 * sum(1 for c in pos_cues if c in text)
        score -= 0.2 * sum(1 for c in neg_cues if c in text)
        if include_target_message:
            score += 0.1  # observed uses explicit target-user statements
        score += self._rng.normal(0.0, 0.08)
        if score >= 0.35:
            return "positive"
        if score <= -0.25:
            return "negative"
        return "neutral"


def sample_instances(*, seed: int = 0, n: int = 32) -> list[ConversationInstance]:
    rng = np.random.default_rng(seed)
    targets = ("DeepSeek", "Claude", "Llama")
    stances = ("negative", "neutral", "positive")
    instances: list[ConversationInstance] = []
    for i in range(n):
        t = targets[int(rng.integers(0, 3))]
        instances.append(
            ConversationInstance(
                instance_id=f"reddit-{i:04d}",
                target_user=f"user_{i}",
                stance_target=t,
                context=f"Thread about {t} capabilities and community reactions. ",
                last_other_message=f"I think {t} shows strong results but trade-offs remain.",
                last_target_message=f"Fair point on {t}; benchmarks look good though.",
                observed_stance=stances[int(rng.integers(0, 3))],
                subreddit="LocalLLaMA",
            )
        )
    return instances


def run_revision_audit(
    instances: list[ConversationInstance],
    strategy: RevisionStrategy,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    sim = ToyStanceSimulator(seed=seed)
    inferred: list[str] = []
    revised: list[str] = []
    tone_orig: list[float] = []
    tone_rev: list[float] = []
    rng = np.random.default_rng(seed)
    for i, inst in enumerate(instances):
        y_inf = sim.infer(inst, include_target_message=False)
        nr = revise_message(inst.last_other_message, strategy, seed=seed + i)
        inst_rev = ConversationInstance(
            instance_id=inst.instance_id,
            target_user=inst.target_user,
            stance_target=inst.stance_target,
            context=inst.context,
            last_other_message=nr,
            last_target_message=inst.last_target_message,
            observed_stance=inst.observed_stance,
            subreddit=inst.subreddit,
        )
        y_rev = simulate_revised_stance(y_inf, strategy, seed=seed + i)
        inferred.append(y_inf)
        revised.append(y_rev)
        t0 = float(rng.uniform(10.0, 90.0))
        tone_orig.append(t0)
        tone_rev.append(revise_tone(t0, strategy))

    pairs = list(zip(inferred, revised, strict=True))
    deltas = [directional_stance_shift(a, b) for a, b in pairs]
    return {
        "strategy": strategy.value,
        "n": len(instances),
        "mean_delta": round(float(np.mean(deltas)), 4),
        "supportive_transition_rate": round(supportive_transition_rate(inferred, revised), 4),
        "sample_revised_message": revise_message(instances[0].last_other_message, strategy, seed=seed),
        "tone_depolarization": depolarization_rate(tone_orig, tone_rev),
    }


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    instances = sample_instances(seed=seed, n=24)
    sim = ToyStanceSimulator(seed=seed)
    observed = [inst.observed_stance for inst in instances]
    inferred = [sim.infer(inst, include_target_message=False) for inst in instances]
    observed_full = [sim.infer(inst, include_target_message=True) for inst in instances]

    stage1 = classification_metrics(observed, inferred)
    add_audit = run_revision_audit(instances, RevisionStrategy.ADD, seed=seed + 1)
    meme_audit = run_revision_audit(instances, RevisionStrategy.MEME, seed=seed + 2)

    pairs_add = [
        (sim.infer(inst, include_target_message=False), simulate_revised_stance(
            sim.infer(inst, include_target_message=False), RevisionStrategy.ADD, seed=seed + i
        ))
        for i, inst in enumerate(instances[:8])
    ]
    return {
        "stage1_macro_f1_toy": stage1["macro_f1"],
        "stage1_accuracy_toy": stage1["accuracy"],
        "add_mean_delta_toy": add_audit["mean_delta"],
        "meme_mean_delta_toy": meme_audit["mean_delta"],
        "add_supportive_rate_toy": add_audit["supportive_transition_rate"],
        "meme_supportive_rate_toy": meme_audit["supportive_transition_rate"],
        "average_shift_add": round(average_directional_shift(pairs_add), 4),
        "observed_vs_inferred_agreement": round(
            sum(1 for a, b in zip(observed, observed_full, strict=True) if a == b) / len(observed),
            3,
        ),
    }
