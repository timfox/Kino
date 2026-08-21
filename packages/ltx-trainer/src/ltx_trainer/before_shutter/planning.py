"""Aesthetic-guided comparative planning (Sec. 3.3)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.before_shutter.camera import exposure_validity_logit, plan_camera_from_graph_constraint
from ltx_trainer.before_shutter.config import BeforeShutterConfig, PortraitPlanState
from ltx_trainer.before_shutter.human import balance_score, collision_penetration, realize_staging
from ltx_trainer.before_shutter.lighting import apply_preset, lighting_ratio, refine_lighting_step
from ltx_trainer.before_shutter.scene_graph import PhotographicSceneGraph, build_demo_graph

import torch


@dataclass
class FrontierEntry:
    state: PortraitPlanState
    score: float
    stage: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AestheticFrontier:
    """F_t = {(s_i, I_i, a_i)} — paper Sec. 3.3."""

    entries: list[FrontierEntry] = field(default_factory=list)
    k: int = 6

    def add_or_replace(self, entry: FrontierEntry) -> None:
        self.entries.append(entry)
        self.entries.sort(key=lambda e: e.score, reverse=True)
        self.entries = self.entries[: self.k]

    def best(self) -> FrontierEntry | None:
        return self.entries[0] if self.entries else None

    def compare_accept(self, candidate_score: float, *, margin: float = 0.05) -> bool:
        best = self.best()
        if best is None:
            return True
        return candidate_score >= best.score - margin


def mock_viewfinder_score(
    state: PortraitPlanState,
    graph: PhotographicSceneGraph,
    *,
    prompt: str,
) -> float:
    """Proxy aesthetic score when MLLM judge unavailable."""
    score = 0.5
    if collision_penetration(state) == 0:
        score += 0.15
    score += 0.2 * balance_score(state)
    if state.preset in prompt.lower() or "dramatic" in prompt.lower():
        score += 0.05
    ratio = lighting_ratio(state.light_powers_w[0], state.light_powers_w[1] if len(state.light_powers_w) > 1 else 1.0)
    if 3.0 <= ratio <= 8.0:
        score += 0.1
    if graph.nodes_non.get("face") and graph.nodes_non["face"].ev100 is not None:
        if abs(graph.nodes_non["face"].ev100) < 2.5:
            score += 0.05
    return min(1.0, score)


def judge_pairwise_preference(score_a: float, score_b: float) -> bool:
    """True if A preferred over B."""
    return score_a >= score_b


def planning_step_staging(
    state: PortraitPlanState,
    graph: PhotographicSceneGraph,
    frontier: AestheticFrontier,
    *,
    prompt: str,
    anchor: str = "red_chair",
) -> tuple[PortraitPlanState, AestheticFrontier, dict[str, Any]]:
    state = realize_staging(state, anchor)
    state.stage = "staging"
    score = mock_viewfinder_score(state, graph, prompt=prompt)
    accepted = frontier.compare_accept(score)
    if accepted:
        frontier.add_or_replace(FrontierEntry(state, score, "staging"))
    return state, frontier, {"accepted": accepted, "score": score, "stage": "staging"}


def planning_step_composition(
    state: PortraitPlanState,
    graph: PhotographicSceneGraph,
    frontier: AestheticFrontier,
    *,
    prompt: str,
) -> tuple[PortraitPlanState, AestheticFrontier, dict[str, Any]]:
    ev_targets = {k: v.ev100 for k, v in graph.nodes_non.items() if v.ev100 is not None}
    state = plan_camera_from_graph_constraint(state, keep_nodes=("face", "hand"), ev_targets=ev_targets)
    state.stage = "composition"
    score = mock_viewfinder_score(state, graph, prompt=prompt) + 0.05
    accepted = frontier.compare_accept(score)
    if accepted:
        frontier.add_or_replace(FrontierEntry(state, score, "composition"))
    return state, frontier, {"accepted": accepted, "score": score, "stage": "composition"}


def planning_step_lighting(
    state: PortraitPlanState,
    graph: PhotographicSceneGraph,
    frontier: AestheticFrontier,
    *,
    prompt: str,
    critique: str = "",
) -> tuple[PortraitPlanState, AestheticFrontier, dict[str, Any]]:
    preset = "chiaroscuro" if "moody" in prompt.lower() or "dramatic" in prompt.lower() else "rembrandt"
    state = apply_preset(state, preset)
    if critique:
        state = refine_lighting_step(state, critique=critique)
    state.stage = "lighting"
    base = torch.full((64, 64), 0.35)
    _, v_exp = exposure_validity_logit(base, cfg=BeforeShutterConfig())
    score = mock_viewfinder_score(state, graph, prompt=prompt) + 0.1 * math.tanh(v_exp)
    accepted = frontier.compare_accept(score)
    if accepted:
        frontier.add_or_replace(FrontierEntry(state, score, "lighting"))
    return state, frontier, {"accepted": accepted, "score": score, "stage": "lighting", "V_exp": v_exp}


def run_comparative_planning(
    *,
    prompt: str = "Graceful pianist, thoughtful",
    cfg: BeforeShutterConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Full staged planning loop stub (staging → composition → lighting)."""
    cfg = cfg or BeforeShutterConfig()
    rng = random.Random(seed)
    graph = build_demo_graph(prompt=prompt)
    state = PortraitPlanState()
    frontier = AestheticFrontier(k=cfg.frontier_size)
    trace: list[dict[str, Any]] = []

    for step in range(cfg.max_staging_steps):
        anchor = rng.choice(["red_chair", "floor", "window"])
        state, frontier, info = planning_step_staging(state, graph, frontier, prompt=prompt, anchor=anchor)
        trace.append(info)

    for step in range(min(3, cfg.max_composition_steps)):
        state, frontier, info = planning_step_composition(state, graph, frontier, prompt=prompt)
        trace.append(info)

    critiques = ["add soft contrast", "avoid wash out", "negative fill on shadow side"]
    for step in range(min(3, cfg.max_lighting_steps)):
        crit = critiques[step % len(critiques)]
        state, frontier, info = planning_step_lighting(state, graph, frontier, prompt=prompt, critique=crit)
        trace.append(info)

    best = frontier.best()
    return {
        "prompt": prompt,
        "graph": graph.to_dict(),
        "frontier_size": len(frontier.entries),
        "best_score": best.score if best else 0.0,
        "best_preset": best.state.preset if best else state.preset,
        "trace_steps": len(trace),
        "accepted_steps": sum(1 for t in trace if t.get("accepted")),
    }
