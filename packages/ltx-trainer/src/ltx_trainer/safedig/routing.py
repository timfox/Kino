"""Robustness-aware pre-training routing (Eq. 9–11)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ltx_trainer.safedig.config import InterventionHook, SafeDIGConfig


def _cos(a: Tensor, b: Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    if a.numel() != b.numel():
        d = min(a.numel(), b.numel())
        a, b = a[:d], b[:d]
    denom = a.norm() * b.norm() + 1e-8
    return float((a @ b / denom).item())


def semantic_score(delta_activations: Tensor, category_embedding: Tensor) -> float:
    """S_sem(ℓ) — Eq. (10)."""
    return _cos(delta_activations.mean(dim=0), category_embedding)


def binding_score(delta_activations: Tensor, latent_embedding: Tensor) -> float:
    """S_bind(ℓ) — Eq. (10)."""
    return _cos(delta_activations.mean(dim=0), latent_embedding)


def stability_score(updates: Tensor) -> float:
    """S_stab(ℓ, o) = 1 - norm(Var[...]) — Eq. (10)."""
    var = updates.var().item()
    return 1.0 - min(1.0, var / (var + 1.0))


def intervention_cost(hook: InterventionHook, operator: str, strength: float) -> float:
    """C(ℓ, o, κ_o) — normalized cost stub — Eq. (41)."""
    dim_norm = math.log1p(hook.dim) / 10.0
    op_cost = 0.1 if operator == "Blend" else 0.15
    return dim_norm + op_cost + abs(strength)


@dataclass
class RoutedPair:
    hook: InterventionHook
    operator: str
    score: float


def robustness_score(
    hook: InterventionHook,
    operator: str,
    *,
    delta_a: Tensor,
    category_emb: Tensor,
    latent_emb: Tensor,
    operator_updates: Tensor,
    strength: float,
    cfg: SafeDIGConfig | None = None,
) -> float:
    """R(ℓ, o) — Eq. (9)."""
    cfg = cfg or SafeDIGConfig()
    alpha, rho, eta, xi = cfg.routing_weights
    s_sem = semantic_score(delta_a, category_emb)
    s_bind = binding_score(delta_a, latent_emb)
    s_stab = stability_score(operator_updates)
    cost = intervention_cost(hook, operator, strength)
    return alpha * s_sem + rho * s_bind + eta * s_stab - xi * cost


def rank_interventions(
    hooks: list[InterventionHook],
    operators: tuple[str, ...],
    *,
    delta_a_by_hook: dict[str, Tensor],
    category_emb: Tensor,
    latent_emb: Tensor,
    strength: float = 0.2,
    cfg: SafeDIGConfig | None = None,
) -> list[RoutedPair]:
    """π = argsort R(ℓ, o) — Eq. (11)."""
    cfg = cfg or SafeDIGConfig()
    pairs: list[RoutedPair] = []
    for hook in hooks:
        delta = delta_a_by_hook.get(hook.name, torch.zeros(64))
        for op in operators:
            updates = delta * (1.2 if op == "Repel" else 0.8)
            score = robustness_score(
                hook,
                op,
                delta_a=delta,
                category_emb=category_emb,
                latent_emb=latent_emb,
                operator_updates=updates,
                strength=strength,
                cfg=cfg,
            )
            pairs.append(RoutedPair(hook, op, score))
    pairs.sort(key=lambda p: p.score, reverse=True)
    return pairs
