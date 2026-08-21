"""SwarmRouter: utility-based task dispatch (Sec. 3.3, eq. 1)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ltx_trainer.swarmharness.config import SwarmHarnessConfig, SwarmNode


@dataclass(frozen=True)
class SwarmTask:
    """Task T requiring skill s (Sec. 3.3)."""

    task_id: str
    skill: str
    submitter_id: str
    credit_pool: float | None = None


def utility_score(
    node: SwarmNode,
    task: SwarmTask,
    *,
    cfg: SwarmHarnessConfig | None = None,
) -> float:
    """
    U(v, T) = w1·1[s∈Sv] + w2·(1−ℓv) + w3·(1−dv/dmax) + w4·τv  (eq. 1).
    """
    cfg = cfg or SwarmHarnessConfig()
    w = cfg.router_weights
    cap = 1.0 if node.capability_match(task.skill) else 0.0
    load_term = 1.0 - max(0.0, min(1.0, node.load_fraction))
    dmax = max(cfg.latency_ceiling_ms, 1e-6)
    lat_term = 1.0 - min(1.0, max(0.0, node.latency_ms) / dmax)
    trust_term = max(0.0, min(1.0, node.trust))
    return (
        w.capability * cap
        + w.load * load_term
        + w.latency * lat_term
        + w.trust * trust_term
    )


def rank_candidates(
    candidates: list[SwarmNode],
    task: SwarmTask,
    *,
    cfg: SwarmHarnessConfig | None = None,
) -> list[tuple[SwarmNode, float]]:
    """Score and sort candidates descending by U(v, T)."""
    cfg = cfg or SwarmHarnessConfig()
    scored = [(n, utility_score(n, task, cfg=cfg)) for n in candidates]
    scored.sort(key=lambda x: (-x[1], x[0].node_id))
    return scored


def route_task(
    candidates: list[SwarmNode],
    task: SwarmTask,
    *,
    cfg: SwarmHarnessConfig | None = None,
    rng: random.Random | None = None,
) -> SwarmNode | None:
    """
    Dispatch to argmax U with random tie-break among top scores (Sec. 3.3).
    Returns None if no candidate has the required skill.
    """
    cfg = cfg or SwarmHarnessConfig()
    rng = rng or random.Random()
    eligible = [n for n in candidates if n.capability_match(task.skill)]
    if not eligible:
        return None
    ranked = rank_candidates(eligible, task, cfg=cfg)
    if not ranked:
        return None
    best_score = ranked[0][1]
    ties = [n for n, s in ranked if abs(s - best_score) < 1e-9]
    return rng.choice(ties)


def route_top_k(
    candidates: list[SwarmNode],
    task: SwarmTask,
    k: int | None = None,
    *,
    cfg: SwarmHarnessConfig | None = None,
) -> list[SwarmNode]:
    """Top-K dispatch for redundant / ensemble tasks (Sec. 3.3)."""
    cfg = cfg or SwarmHarnessConfig()
    k = k if k is not None else cfg.top_k_redundant
    eligible = [n for n in candidates if n.capability_match(task.skill)]
    ranked = [n for n, _ in rank_candidates(eligible, task, cfg=cfg)]
    return ranked[: max(1, k)]


def routing_report(
    candidates: list[SwarmNode],
    task: SwarmTask,
    *,
    cfg: SwarmHarnessConfig | None = None,
) -> dict[str, Any]:
    ranked = rank_candidates(candidates, task, cfg=cfg)
    return {
        "task_id": task.task_id,
        "skill": task.skill,
        "candidates": [
            {"node_id": n.node_id, "utility": round(s, 6), "trust": n.trust, "load": n.load_fraction}
            for n, s in ranked
        ],
        "selected": route_task(candidates, task, cfg=cfg).node_id if ranked else None,
    }
