"""Routed agentic workflow stub (LangGraph-style state graph)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ltx_trainer.cbp_assist.ontology import (
    ProvidedCapability,
    RequiredCapability,
    default_mps500_model,
    required_drill_at_station,
)
from ltx_trainer.cbp_assist.smt_planner import PlanningResult, plan_capabilities

Intent = Literal["knowledge_query", "planning_request", "runtime_failure"]


@dataclass
class SharedState:
    user_message: str
    intent: Intent | None = None
    hitl_pending: str | None = None
    required: RequiredCapability | None = None
    provided: dict[str, ProvidedCapability] = field(default_factory=dict)
    planning: PlanningResult | None = None
    natural_language_reply: str = ""
    adaptation_proposal: dict[str, Any] | None = None
    trace: list[str] = field(default_factory=list)


def classify_intent(message: str) -> Intent:
    low = message.lower()
    if any(w in low for w in ("defective", "broken", "failed", "unavailable", "offline")):
        return "runtime_failure"
    if any(w in low for w in ("plan", "produce", "drill", "transport", "manufacture")):
        return "planning_request"
    return "knowledge_query"


def run_workflow(
    message: str,
    *,
    auto_approve_hitl: bool = True,
    apply_adaptation: bool = True,
) -> SharedState:
    """Execute router → specialists with fixed HitL gates (demo)."""
    model = default_mps500_model()
    state = SharedState(user_message=message, provided=dict(model["provided"]))
    state.intent = classify_intent(message)
    state.trace.append(f"router:{state.intent}")

    if state.intent == "knowledge_query":
        state.trace.append("knowledge_retrieval:sparql_select")
        caps = list(state.provided.values())
        names = ", ".join(c.resource for c in caps)
        state.natural_language_reply = (
            f"Available resources: {names}. "
            f"Drilling supports depth 5–10 mm at station 3; transport connects stations 3 and 7."
        )
        return state

    if state.intent == "runtime_failure":
        state.trace.append("analyze_adaptation:runtime_report")
        state.trace.append("capability_mapper:identify_affected")
        if "conveyor" in message.lower():
            state.adaptation_proposal = {
                "action": "remove_capability",
                "capability_id": "cap_conveyor",
                "reason": "Conveyor reported defective",
            }
            state.hitl_pending = "approve_runtime_capability_update"
            if auto_approve_hitl and apply_adaptation:
                state.provided.pop("cap_conveyor", None)
                state.trace.append("repair:sparql_delete")
                state.hitl_pending = None
        state.trace.append("planning:smt_invoke")
        state.required = required_drill_at_station(7.0, 3)
        state.planning = plan_capabilities(
            state.required, state.provided, conveyor_available=False
        )
        state.natural_language_reply = _explain_plan(state.planning)
        return state

    # planning_request
    state.trace.append("knowledge_retrieval:required_capability_candidates")
    depth = 2.0 if "2 mm" in message or "2mm" in message else 7.0
    station = 15 if "station 15" in message.lower() or "station_id 15" in message else 3
    state.required = required_drill_at_station(depth, station)
    state.hitl_pending = "confirm_required_capability"
    state.trace.append("hitl:confirm_required_capability")

    if auto_approve_hitl:
        state.hitl_pending = None
        state.trace.append("planning:smt_invoke")
        state.planning = plan_capabilities(state.required, state.provided)

        if state.planning.satisfiable:
            state.trace.append("result_interpretation:explain_sat")
            state.natural_language_reply = _explain_plan(state.planning)
            return state

        state.trace.append("analyze_adaptation:unsat_core_cot")
        state.trace.append("capability_mapper:map_constraints")
        state.adaptation_proposal = _propose_adaptation(state.planning, state.required)
        state.hitl_pending = "approve_model_adaptation"
        state.trace.append("hitl:approve_adaptation")

        if auto_approve_hitl and apply_adaptation and state.adaptation_proposal:
            _apply_proposal(state, state.adaptation_proposal)
            state.trace.append("repair:sparql_update")
            state.hitl_pending = None
            state.trace.append("planning:replan")
            state.required = required_drill_at_station(
                state.adaptation_proposal.get("depth_mm", 7.0),
                state.adaptation_proposal.get("station_id", 3),
            )
            state.planning = plan_capabilities(state.required, state.provided)
            state.natural_language_reply = _explain_plan(state.planning)
        else:
            state.natural_language_reply = _explain_unsat(state.planning)
    return state


def _propose_adaptation(result: PlanningResult, req: RequiredCapability) -> dict[str, Any]:
    depth = float(req.constraints.get("depth_mm", 7))
    station = int(req.constraints.get("station_id", 3))
    fixes: dict[str, Any] = {}
    for c in result.unsat_core:
        if "depth_mm" in c:
            fixes["depth_mm"] = 5.0
        if "station_id" in c:
            fixes["station_id"] = 3
    if not fixes:
        fixes = {"depth_mm": max(5.0, depth), "station_id": 3}
    return {
        **fixes,
        "rationale": "Align required depth and station with drilling module constraints",
        "conflicts": result.unsat_core,
    }


def _apply_proposal(state: SharedState, proposal: dict[str, Any]) -> None:
    if "depth_mm" in proposal:
        state.required = required_drill_at_station(
            float(proposal["depth_mm"]), int(proposal.get("station_id", 3))
        )


def _explain_plan(result: PlanningResult) -> str:
    if not result.satisfiable:
        return _explain_unsat(result)
    lines = ["Satisfiable plan:"]
    for i, step in enumerate(result.plan, 1):
        params = ", ".join(f"{k}={v}" for k, v in step.parameters.items())
        lines.append(f"{i}. {step.capability_id} ({step.resource}): {params}")
    return "\n".join(lines)


def _explain_unsat(result: PlanningResult) -> str:
    if result.satisfiable:
        return _explain_plan(result)
    parts = ["Planning infeasible under current capability model:"]
    for i, c in enumerate(result.unsat_core, 1):
        parts.append(f"Conflict {i}: {c}")
    return "\n".join(parts)
