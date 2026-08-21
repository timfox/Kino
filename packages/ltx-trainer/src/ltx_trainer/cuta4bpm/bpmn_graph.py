"""Minimal BPMN-like graph (nodes + sequence flows) for transformation smoke."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

BpmnNodeKind = Literal[
    "start_event",
    "end_event",
    "task",
    "exclusive_gateway",
    "parallel_gateway",
    "inclusive_gateway",
    "pool",
    "lane",
]


@dataclass
class BpmnNode:
    id: str
    kind: BpmnNodeKind
    label: str = ""
    lane: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BpmnFlow:
    id: str
    source: str
    target: str
    condition: str = ""


@dataclass
class BpmnGraph:
    """Flattened BPMN subgraph container."""

    nodes: list[BpmnNode] = field(default_factory=list)
    flows: list[BpmnFlow] = field(default_factory=list)
    pool: str = ""

    def add_node(self, kind: BpmnNodeKind, label: str = "", **meta: Any) -> str:
        nid = f"n{len(self.nodes)}"
        self.nodes.append(BpmnNode(id=nid, kind=kind, label=label, metadata=dict(meta)))
        return nid

    def connect(self, source: str, target: str, condition: str = "") -> None:
        fid = f"f{len(self.flows)}"
        self.flows.append(BpmnFlow(id=fid, source=source, target=target, condition=condition))

    def summary(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for n in self.nodes:
            kinds[n.kind] = kinds.get(n.kind, 0) + 1
        return {
            "node_count": len(self.nodes),
            "flow_count": len(self.flows),
            "node_kinds": kinds,
            "pool": self.pool,
        }
