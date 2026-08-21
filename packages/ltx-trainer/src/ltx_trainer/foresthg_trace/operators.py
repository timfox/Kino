"""Deterministic operators O over scene hypergraphs (Section 3.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ltx_trainer.foresthg_trace.hypergraph import SceneHypergraph, TreeNode

OperatorName = Literal["read", "filter", "expand", "aggregate", "compare", "audit", "answer"]


@dataclass
class ReasoningContext:
    selected_ids: set[int] = field(default_factory=set)
    groups: dict[str, list[int]] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class OperatorCall:
    op: OperatorName
    args: dict[str, Any]


@dataclass
class StepResult:
    context: ReasoningContext
    payload: dict[str, Any]


def _poor_health(n: TreeNode, threshold: float = 0.5) -> bool:
    return n.health < threshold


def execute_operator(
    graph: SceneHypergraph,
    ctx: ReasoningContext,
    call: OperatorCall,
) -> StepResult:
    op, args = call.op, call.args
    ctx = ReasoningContext(
        selected_ids=set(ctx.selected_ids),
        groups=dict(ctx.groups),
        scalars=dict(ctx.scalars),
        records=list(ctx.records) + [{"op": op, "args": args}],
    )

    if op == "read":
        scope = str(args.get("scope", "all_trees"))
        if scope == "all_trees":
            ctx.selected_ids = set(graph.nodes.keys())
        elif scope == "overstory":
            heights = [n.height_m for n in graph.node_list()]
            cutoff = sorted(heights)[-max(1, len(heights) // 3)] if heights else 0.0
            ctx.selected_ids = {n.node_id for n in graph.node_list() if n.height_m >= cutoff}
        payload = {"count": len(ctx.selected_ids), "scope": scope}
        return StepResult(ctx, payload)

    if op == "filter":
        field_name = str(args.get("field", "slope_class"))
        value = args.get("value")
        health_lt = args.get("health_lt")
        pool = [graph.nodes[i] for i in ctx.selected_ids if i in graph.nodes] or graph.node_list()
        kept: list[int] = []
        for n in pool:
            if value is not None and getattr(n, field_name, None) != value:
                continue
            if health_lt is not None and not _poor_health(n, float(health_lt)):
                continue
            kept.append(n.node_id)
        key = str(args.get("group_key", f"{field_name}={value}"))
        ctx.groups[key] = kept
        ctx.selected_ids = set(kept)
        payload = {"group_key": key, "count": len(kept)}
        return StepResult(ctx, payload)

    if op == "expand":
        radius = float(args.get("radius_m", 25.0))
        seed = [graph.nodes[i] for i in ctx.selected_ids if i in graph.nodes]
        if not seed:
            seed = graph.node_list()[:1]
        expanded: set[int] = set(ctx.selected_ids)
        for s in seed:
            for n in graph.node_list():
                d = ((n.x - s.x) ** 2 + (n.y - s.y) ** 2) ** 0.5
                if d <= radius:
                    expanded.add(n.node_id)
        ctx.selected_ids = expanded
        payload = {"expanded_count": len(expanded), "radius_m": radius}
        return StepResult(ctx, payload)

    if op == "aggregate":
        metric = str(args.get("metric", "mean_health"))
        pool = [graph.nodes[i] for i in ctx.selected_ids if i in graph.nodes]
        if metric == "mean_health":
            val = sum(n.health for n in pool) / max(1, len(pool))
        elif metric == "count":
            val = float(len(pool))
        elif metric == "poor_health_ratio":
            val = sum(1 for n in pool if _poor_health(n)) / max(1, len(pool))
        else:
            val = 0.0
        key = str(args.get("store_as", metric))
        ctx.scalars[key] = val
        payload = {"metric": metric, "value": val, "n": len(pool)}
        return StepResult(ctx, payload)

    if op == "compare":
        a_key = str(args.get("a"))
        b_key = str(args.get("b"))
        a = ctx.scalars.get(a_key, 0.0)
        b = ctx.scalars.get(b_key, 0.0)
        winner = a_key if a > b else b_key if b > a else "tie"
        ctx.scalars["compare_winner"] = float(1.0 if winner == a_key else 0.0 if winner == b_key else 0.5)
        payload = {"a": a, "b": b, "winner": winner}
        return StepResult(ctx, payload)

    if op == "audit":
        payload = {
            "selected": len(ctx.selected_ids),
            "groups": {k: len(v) for k, v in ctx.groups.items()},
            "scalars": dict(ctx.scalars),
        }
        return StepResult(ctx, payload)

    if op == "answer":
        template = str(args.get("template", "numeric"))
        if template == "slope_ratio":
            text = "Shady slope has a higher poor-health tree ratio."
        elif template == "degraded_overstory":
            text = "Degraded overstory: poor-health overstory with suppressed canopy."
        else:
            text = str(args.get("text", "unknown"))
        payload = {"answer": text, "template": template}
        return StepResult(ctx, payload)

    raise ValueError(f"unknown operator {op!r}")
