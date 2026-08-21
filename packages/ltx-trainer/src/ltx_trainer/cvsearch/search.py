"""Assess-then-Search workflow and Dynamic Bottom-Up Search (Sec. 4, Algorithm 1/3)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.cvsearch.config import CVSearchConfig, SearchMode
from ltx_trainer.cvsearch.sgap import TreeNode


def information_sufficiency(confidence_yes: float) -> float:
    """cq(I) = M('Yes' | pq(Q), I) (Eq. 2); stub uses precomputed confidence."""
    return max(0.0, min(1.0, confidence_yes))


def route_search_mode(
    cq_global: float,
    expert_proposals_nonempty: bool,
    expert_coverage_ok: bool,
    *,
    tau_q: float,
) -> SearchMode:
    """Cognitive routing: direct → expert → scan (Fig. 2a)."""
    if cq_global >= tau_q:
        return SearchMode.DIRECT_ANSWER
    if expert_proposals_nonempty and expert_coverage_ok:
        return SearchMode.EXPERT_SEARCH
    return SearchMode.SCAN_SEARCH


def expert_coverage_valid(num_proposals: int, num_targets: int) -> bool:
    """Coverage when segmented categories strictly match extracted targets (Sec. 4.2.2)."""
    return num_proposals > 0 and num_proposals >= num_targets


def node_priority(
    cv: float,
    co: float,
    child_priority_max: float,
    *,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    """cx = α·cv + β·co + γ·c*x (Eq. 5)."""
    return alpha * cv + beta * co + gamma * child_priority_max


def dynamic_threshold(step: int, *, tau_q: float, tau_q_hat: float, delta_tau: float) -> float:
    """τ_curr = max(τq − k·Δτ, τ̂q) (Appendix Algorithm 3)."""
    return max(tau_q - step * delta_tau, tau_q_hat)


@dataclass
class BottomUpResult:
    found: bool
    candidate_node_id: str | None
    sufficiency: float


def bottom_up_search_layer(
    nodes: list[TreeNode],
    existence_confidences: dict[str, float],
    *,
    cfg: CVSearchConfig | None = None,
    child_priorities: dict[str, float] | None = None,
    start_step: int = 0,
) -> tuple[BottomUpResult, int]:
    """One layer of bottom-up search; returns best candidate or FOUND node."""
    cfg = cfg or CVSearchConfig()
    child_priorities = child_priorities or {}
    ranked: list[tuple[TreeNode, float]] = []
    for n in nodes:
        if n.pruned:
            continue
        c_star = child_priorities.get(n.node_id, 0.0)
        co = existence_confidences.get(n.node_id, 0.0)
        cx = node_priority(
            n.visual_complexity,
            co,
            c_star,
            alpha=cfg.alpha_cv,
            beta=cfg.beta_co,
            gamma=cfg.gamma_child,
        )
        ranked.append((n, cx))
    ranked.sort(key=lambda x: x[1], reverse=True)

    step = start_step
    for node, _ in ranked:
        cq = information_sufficiency(existence_confidences.get(node.node_id, 0.0))
        tau_curr = dynamic_threshold(step, tau_q=cfg.tau_q, tau_q_hat=cfg.tau_q_hat, delta_tau=cfg.delta_tau)
        if cq > tau_curr:
            return BottomUpResult(True, node.node_id, cq), step
        step += 1

    top = ranked[0][0] if ranked else None
    return BottomUpResult(False, top.node_id if top else None, 0.0), step
