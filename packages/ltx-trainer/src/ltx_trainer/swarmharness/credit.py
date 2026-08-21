"""SwarmCredit: Shapley attribution, trust decay, genesis (Sec. 4)."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ltx_trainer.swarmharness.config import SwarmHarnessConfig


QualityFn = Callable[[frozenset[str]], float]


def leave_one_out_quality_proxy(
    coalition: frozenset[str],
    *,
    node_weights: dict[str, float],
    baseline: float = 0.0,
) -> float:
    """
    Cheap q(S) proxy (Sec. 4.3): coalition quality ≈ baseline + Σ weights in S,
    capped at 1. Used when per-node marginal data is unavailable.
    """
    if not coalition:
        return baseline
    gain = sum(node_weights.get(n, 0.0) for n in coalition)
    return max(0.0, min(1.0, baseline + gain))


def shapley_monte_carlo(
    nodes: list[str],
    quality_fn: QualityFn,
    *,
    samples: int = 100,
    rng: random.Random | None = None,
) -> dict[str, float]:
    """Algorithm 1, Step 1 — permutation sampling Shapley estimates."""
    if not nodes:
        return {}
    rng = rng or random.Random()
    phi = {n: 0.0 for n in nodes}
    k = len(nodes)
    for _ in range(samples):
        perm = nodes[:]
        rng.shuffle(perm)
        prefix: list[str] = []
        q_prev = quality_fn(frozenset())
        for node in perm:
            prefix.append(node)
            q_now = quality_fn(frozenset(prefix))
            phi[node] += q_now - q_prev
            q_prev = q_now
    inv_m = 1.0 / float(samples)
    return {n: phi[n] * inv_m for n in nodes}


def allocate_credits(
    phi: dict[str, float],
    credit_pool: float,
) -> dict[str, float]:
    """Algorithm 1, Step 2 — normalise positive Shapley values to C(T)."""
    if credit_pool <= 0:
        return {n: 0.0 for n in phi}
    positive = {n: max(v, 0.0) for n, v in phi.items()}
    total = sum(positive.values())
    if total > 0:
        return {n: (positive[n] / total) * credit_pool for n in phi}
    k = len(phi)
    if k == 0:
        return {}
    uniform = credit_pool / k
    return {n: uniform for n in phi}


def update_trust_scores(
    trust: dict[str, float],
    deltas: dict[str, float],
    *,
    alpha: float,
) -> dict[str, float]:
    """Algorithm 1, Step 3 — trust learning from credit attribution."""
    out = dict(trust)
    for node_id, delta in deltas.items():
        tau = out.get(node_id, 0.0)
        if delta > 0:
            out[node_id] = tau + alpha * (1.0 - tau)
        else:
            out[node_id] = tau - alpha * tau
        out[node_id] = max(0.0, min(1.0, out[node_id]))
    return out


def trust_decay(
    trust: float,
    dt_seconds: float,
    *,
    beta: float,
    period_seconds: float,
) -> float:
    """τv ← τv · (1 − β)^(Δt/T0)  (eq. 2)."""
    if period_seconds <= 0 or dt_seconds <= 0:
        return trust
    exponent = dt_seconds / period_seconds
    factor = (1.0 - beta) ** exponent
    return max(0.0, min(1.0, trust * factor))


@dataclass
class AttributionResult:
    """Output of SwarmCredit attribution for one completed task."""

    deltas: dict[str, float]
    trust: dict[str, float]
    shapley_raw: dict[str, float]
    submitter_balance: float
    credit_pool: float


def swarm_credit_attribution(
    nodes: list[str],
    quality_fn: QualityFn,
    credit_pool: float,
    trust: dict[str, float],
    submitter_balance: float,
    *,
    cfg: SwarmHarnessConfig | None = None,
    rng: random.Random | None = None,
) -> AttributionResult:
    """
    Full Algorithm 1: Shapley sample → normalise → trust update → submitter debit.
    """
    cfg = cfg or SwarmHarnessConfig()
    phi = shapley_monte_carlo(nodes, quality_fn, samples=cfg.shapley_samples, rng=rng)
    deltas = allocate_credits(phi, credit_pool)
    new_trust = update_trust_scores(trust, deltas, alpha=cfg.trust_learning_rate)
    new_balance = submitter_balance - credit_pool
    return AttributionResult(
        deltas=deltas,
        trust=new_trust,
        shapley_raw=phi,
        submitter_balance=new_balance,
        credit_pool=credit_pool,
    )


def single_node_attribution(
    node_id: str,
    quality: float,
    credit_pool: float,
    trust: dict[str, float],
    submitter_balance: float,
    *,
    cfg: SwarmHarnessConfig | None = None,
) -> AttributionResult:
    """k=1 exact attribution (Sec. 4.6): O(1) cost."""
    cfg = cfg or SwarmHarnessConfig()

    def q(coalition: frozenset[str]) -> float:
        return quality if node_id in coalition else 0.0

    return swarm_credit_attribution(
        [node_id],
        q,
        credit_pool,
        trust,
        submitter_balance,
        cfg=cfg,
        rng=random.Random(0),
    )


@dataclass
class GenesisState:
    """Proof-of-contribution genesis (Sec. 4.5)."""

    credit: float = 0.0
    locked: bool = True
    served_proof: bool = False
    free_task_used: bool = False


def grant_node_genesis(state: GenesisState, *, cfg: SwarmHarnessConfig | None = None) -> GenesisState:
    cfg = cfg or SwarmHarnessConfig()
    return GenesisState(
        credit=cfg.genesis_credit,
        locked=True,
        served_proof=state.served_proof,
        free_task_used=state.free_task_used,
    )


def unlock_genesis_after_serve(state: GenesisState) -> GenesisState:
    """Countersigned serve unlocks locked genesis pool."""
    return GenesisState(
        credit=state.credit,
        locked=False,
        served_proof=True,
        free_task_used=state.free_task_used,
    )


def grant_submitter_genesis_task(state: GenesisState, *, cfg: SwarmHarnessConfig | None = None) -> GenesisState:
    cfg = cfg or SwarmHarnessConfig()
    if state.free_task_used:
        return state
    return GenesisState(
        credit=state.credit + cfg.genesis_task_grant,
        locked=state.locked,
        served_proof=state.served_proof,
        free_task_used=True,
    )


def shapley_standard_error(samples: int) -> float:
    """O(1/√M) convergence bound (Sec. 4.6)."""
    if samples <= 0:
        return float("inf")
    return 1.0 / math.sqrt(samples)


def attribution_summary(result: AttributionResult) -> dict[str, Any]:
    total = sum(result.deltas.values())
    return {
        "credit_pool": result.credit_pool,
        "deltas": {k: round(v, 6) for k, v in result.deltas.items()},
        "shapley_raw": {k: round(v, 6) for k, v in result.shapley_raw.items()},
        "efficiency_error": abs(total - result.credit_pool),
        "submitter_balance": round(result.submitter_balance, 6),
    }
