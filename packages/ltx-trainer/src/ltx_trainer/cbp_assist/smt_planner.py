"""Symbolic SMT planning stub — formal correctness stays here, not in LLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.cbp_assist.ontology import ProvidedCapability, RequiredCapability


@dataclass
class PlanStep:
    capability_id: str
    resource: str
    parameters: dict[str, Any]


@dataclass
class PlanningResult:
    satisfiable: bool
    plan: list[PlanStep] = field(default_factory=list)
    unsat_core: list[str] = field(default_factory=list)
    explanation_struct: dict[str, Any] = field(default_factory=dict)


def _check_drill(req: RequiredCapability, cap: ProvidedCapability) -> list[str]:
    conflicts: list[str] = []
    depth = float(req.constraints.get("depth_mm", 0))
    lo, hi = cap.constraints.get("depth_mm", (0, 0))
    if depth < lo or depth > hi:
        conflicts.append(
            f"depth_mm={depth} outside drill range [{lo}, {hi}] for {cap.id}"
        )
    req_station = req.constraints.get("station_id")
    cap_station = cap.constraints.get("station_id")
    if req_station is not None and cap_station is not None and req_station != cap_station:
        conflicts.append(
            f"station_id={req_station} != provided station_id={cap_station} for {cap.id}"
        )
    return conflicts


def plan_capabilities(
    required: RequiredCapability,
    provided: dict[str, ProvidedCapability],
    *,
    conveyor_available: bool = True,
) -> PlanningResult:
    """Toy planner: transport (if stations differ) + drill when constraints align."""
    active = {
        k: v
        for k, v in provided.items()
        if not (k == "cap_conveyor" and not conveyor_available)
    }
    drill = active.get("cap_drill")
    transport = active.get("cap_transport")
    if drill is None:
        return PlanningResult(
            satisfiable=False,
            unsat_core=["no drilling capability available"],
            explanation_struct={"missing": "cap_drill"},
        )

    conflicts = _check_drill(required, drill)
    req_station = required.constraints.get("station_id", 3)
    drill_station = drill.constraints.get("station_id", 3)

    if conflicts:
        return PlanningResult(
            satisfiable=False,
            unsat_core=conflicts,
            explanation_struct={
                "required": required.constraints,
                "drill_constraints": drill.constraints,
            },
        )

    steps: list[PlanStep] = []
    if transport is not None and req_station != drill_station:
        end_stations = transport.constraints.get("station_id", {drill_station})
        target = req_station if req_station in end_stations else drill_station
        steps.append(
            PlanStep(
                capability_id=transport.id,
                resource=transport.resource,
                parameters={"from_station": drill_station, "to_station": target},
            )
        )

    steps.append(
        PlanStep(
            capability_id=drill.id,
            resource=drill.resource,
            parameters={
                "depth_mm": required.constraints.get("depth_mm"),
                "station_id": drill_station,
            },
        )
    )
    return PlanningResult(satisfiable=True, plan=steps)
