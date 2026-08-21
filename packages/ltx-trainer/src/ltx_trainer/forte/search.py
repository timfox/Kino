"""Best-first beam search for FOL query refinement (arXiv:2606.05812 §3.1)."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from ltx_trainer.forte.fol import (
    FolForm,
    RefinementOperator,
    apply_operator,
    invariant_set,
    parse_query_fallback,
    pred_set,
    verbalise,
)


def _hash_embed(text: str, dim: int = 512) -> np.ndarray:
    h = hashlib.sha256(text.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    v = rng.standard_normal(dim)
    n = np.linalg.norm(v)
    return v / (n + 1e-8)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


@dataclass
class SearchConfig:
    beam_width: int = 5
    max_depth: int = 4
    lambda_neg: float = 1.0
    beta_pivot: float = 0.5
    tau: float = 0.0
    embed_dim: int = 512


@dataclass
class RefinementResult:
    phi0: FolForm
    phi_star: FolForm
    q_star: str
    score: float
    explored: int


def generate_elaborations(query: str) -> tuple[FolForm, FolForm, FolForm]:
    """Toy q+, q- from query keywords (no LLM)."""
    phi0 = parse_query_fallback(query)
    phi_plus = phi0.copy()
    phi_minus = phi0.copy()
    if "Bird" in phi0.predicates:
        phi_plus.predicates.update({"Peaceful", "Outdoor", "Soft"})
        phi_minus.predicates.update({"DistressCall", "HighIntensity"})
    elif "Speaking" in phi0.predicates:
        phi_plus.predicates.add("Quiet")
        phi_minus.negated.add("Quiet")
        phi_minus.predicates.add("Shouting")
    elif "Footsteps" in phi0.predicates:
        phi_plus.predicates.update({"Quiet", "Empty"})
        phi_minus.predicates.add("Loud")
    elif "Rain" in phi0.predicates:
        phi_plus.predicates.add("Metallic")
        phi_minus.predicates.add("Organic")
    else:
        phi_plus.predicates.add("Detailed")
        phi_minus.predicates.add("Noisy")
    return phi0, phi_plus, phi_minus


def pivot_direction(phi_plus: FolForm, phi_minus: FolForm, *, dim: int = 512) -> np.ndarray:
    e_plus = _hash_embed(verbalise(phi_plus), dim)
    e_minus = _hash_embed(verbalise(phi_minus), dim)
    v = e_plus - e_minus
    n = np.linalg.norm(v)
    return v / (n + 1e-8) if n > 1e-8 else e_plus


def operator_candidates(phi: FolForm, depth: int, invariant: set[str]) -> list[tuple[RefinementOperator, str]]:
    """Depth-scheduled round-robin: attr @1, rel @2, neg @3-4."""
    cands: list[tuple[RefinementOperator, str]] = []
    if depth == 1:
        for p in ("Quiet", "Peaceful", "Soft", "Metallic", "Empty"):
            if p not in phi.predicates:
                cands.append((RefinementOperator.ATTR, p))
    elif depth == 2:
        for r in ("In", "On", "Background"):
            cands.append((RefinementOperator.REL, r))
    else:
        for p in ("Shouting", "DistressCall", "AlarmCall", "Loud", "Crowd"):
            if p not in phi.negated:
                cands.append((RefinementOperator.NEG, p))
    # retain invariant
    filtered = []
    for op, arg in cands:
        trial = apply_operator(phi, op, arg)
        if trial is None:
            continue
        if invariant <= pred_set(trial):
            filtered.append((op, arg))
    return filtered[:5]


def objective(
    phi: FolForm,
    *,
    e_plus_a: np.ndarray,
    e_minus_a: np.ndarray,
    pivot: np.ndarray,
    lambda_neg: float,
    beta: float,
    dim: int,
) -> float:
    text = verbalise(phi)
    e_q = _hash_embed(text, dim)
    u = cosine(e_q, e_plus_a) - lambda_neg * cosine(e_q, e_minus_a) + beta * cosine(e_q, pivot)
    c = len(pred_set(phi))
    return u - c


def feasible(phi: FolForm, pivot: np.ndarray, tau: float, *, dim: int) -> bool:
    e_q = _hash_embed(verbalise(phi), dim)
    return cosine(e_q, pivot) >= tau


def best_first_search(
    query: str,
    *,
    e_plus_a: np.ndarray | None = None,
    e_minus_a: np.ndarray | None = None,
    cfg: SearchConfig | None = None,
) -> RefinementResult:
    cfg = cfg or SearchConfig()
    phi0, phi_plus, phi_minus = generate_elaborations(query)
    inv = invariant_set(phi0, phi_plus, phi_minus)
    pivot = pivot_direction(phi_plus, phi_minus, dim=cfg.embed_dim)
    if e_plus_a is None:
        e_plus_a = _hash_embed(verbalise(phi_plus), cfg.embed_dim)
    if e_minus_a is None:
        e_minus_a = _hash_embed(verbalise(phi_minus), cfg.embed_dim)

    frontier: list[tuple[float, FolForm]] = [(0.0, phi0)]
    best = phi0
    best_score = objective(
        phi0,
        e_plus_a=e_plus_a,
        e_minus_a=e_minus_a,
        pivot=pivot,
        lambda_neg=cfg.lambda_neg,
        beta=cfg.beta_pivot,
        dim=cfg.embed_dim,
    )
    explored = 0

    for depth in range(1, cfg.max_depth + 1):
        next_frontier: list[tuple[float, FolForm]] = []
        for _, phi in frontier:
            for op, arg in operator_candidates(phi, depth, inv):
                child = apply_operator(phi, op, arg)
                if child is None or not feasible(child, pivot, cfg.tau, dim=cfg.embed_dim):
                    continue
                explored += 1
                f = objective(
                    child,
                    e_plus_a=e_plus_a,
                    e_minus_a=e_minus_a,
                    pivot=pivot,
                    lambda_neg=cfg.lambda_neg,
                    beta=cfg.beta_pivot,
                    dim=cfg.embed_dim,
                )
                if f > best_score:
                    best_score = f
                    best = child
                next_frontier.append((f, child))
        next_frontier.sort(key=lambda x: -x[0])
        frontier = next_frontier[: cfg.beam_width]
        if not frontier:
            break

    q_star = verbalise(best)
    return RefinementResult(phi0=phi0, phi_star=best, q_star=q_star, score=best_score, explored=explored)
