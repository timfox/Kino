"""Toy MPS500-style capability ontology for demos (CSS reference model stub)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CapabilityParam:
    name: str
    value: Any
    unit: str = ""


@dataclass
class ProvidedCapability:
    id: str
    resource: str
    description: str
    inputs: list[CapabilityParam] = field(default_factory=list)
    outputs: list[CapabilityParam] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequiredCapability:
    id: str
    description: str
    outputs: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)


def default_mps500_model() -> dict[str, Any]:
    """Laboratory modular production system capabilities (paper evaluation)."""
    provided = [
        ProvidedCapability(
            id="cap_transport",
            resource="mobile_robot",
            description="Transport workpiece between stations",
            constraints={"station_id": {3, 7}, "reachable": True},
        ),
        ProvidedCapability(
            id="cap_drill",
            resource="drilling_module",
            description="Drill hole at station",
            constraints={"depth_mm": (5.0, 10.0), "station_id": 3},
        ),
        ProvidedCapability(
            id="cap_conveyor",
            resource="conveyor",
            description="Convey workpiece along line",
            constraints={"station_id": {3, 7}, "available": True},
        ),
    ]
    return {
        "provided": {c.id: c for c in provided},
        "resources": ["mobile_robot", "drilling_module", "conveyor"],
    }


def required_drill_at_station(depth_mm: float, station_id: int) -> RequiredCapability:
    return RequiredCapability(
        id="req_finish_part",
        description="Finish part with drilled hole at target station",
        outputs={"depth_mm": depth_mm, "station_id": station_id},
        constraints={"depth_mm": depth_mm, "station_id": station_id},
    )
