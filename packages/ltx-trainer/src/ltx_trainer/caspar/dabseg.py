"""Directed Acyclic Bipartite Symbolic Expression Graph (DABSEG)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeKind(str, Enum):
    CALL = "call"
    VALUE = "value"


class CallOp(str, Enum):
    ADD = "add"
    MUL = "mul"
    DIV = "div"
    SIN = "sin"
    COS = "cos"
    NORM3 = "norm3"
    RNORM3 = "rnorm3"
    FMA = "fma"
    READ = "read"
    WRITE = "write"
    CONTRIBUTE = "contribute"
    ACCUM_SUM = "accum_sum"


@dataclass
class DabsegNode:
    id: int
    kind: NodeKind
    op: CallOp | None = None
    inputs: list[int] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DabsegGraph:
    """Bipartite graph: call nodes invoke ops; value nodes hold register bindings."""

    nodes: dict[int, DabsegNode] = field(default_factory=dict)
    _next_id: int = 0

    def new_value(self, *, source_call: int | None = None) -> int:
        nid = self._next_id
        self._next_id += 1
        self.nodes[nid] = DabsegNode(id=nid, kind=NodeKind.VALUE, meta={"source_call": source_call})
        return nid

    def new_call(self, op: CallOp, inputs: list[int], *, meta: dict[str, Any] | None = None) -> int:
        nid = self._next_id
        self._next_id += 1
        outs = [self.new_value(source_call=nid) for _ in range(meta.get("n_outputs", 1) if meta else 1)]
        self.nodes[nid] = DabsegNode(
            id=nid,
            kind=NodeKind.CALL,
            op=op,
            inputs=inputs,
            outputs=outs,
            meta=meta or {},
        )
        return nid

    def call_nodes(self) -> list[DabsegNode]:
        return [n for n in self.nodes.values() if n.kind is NodeKind.CALL]

    def value_nodes(self) -> list[DabsegNode]:
        return [n for n in self.nodes.values() if n.kind is NodeKind.VALUE]

    def dependents(self, value_id: int) -> list[int]:
        """Calls that consume a given value."""
        out: list[int] = []
        for n in self.call_nodes():
            if value_id in n.inputs:
                out.append(n.id)
        return out


def build_sin_cos_sum_graph(x: str = "x1", x0: str = "x0") -> DabsegGraph:
    """
    Paper Fig. 1 example: [sin(x1)+x0+x1+1, cos(x1)-1].
    Returns a minimal DABSEG with call/value nodes (names stored in meta).
    """
    g = DabsegGraph()
    v_x0 = g.new_value()
    g.nodes[v_x0].meta["symbol"] = x0
    v_x1 = g.new_value()
    g.nodes[v_x1].meta["symbol"] = x

    c_sin = g.new_call(CallOp.SIN, [v_x1], meta={"symbol": f"sin({x})"})
    v_sin = g.nodes[c_sin].outputs[0]
    c_cos = g.new_call(CallOp.COS, [v_x1], meta={"symbol": f"cos({x})"})
    v_cos = g.nodes[c_cos].outputs[0]

    c_sum = g.new_call(CallOp.ACCUM_SUM, [v_sin, v_x0, v_x1], meta={"expr": "sin(x1)+x0+x1+1", "n_outputs": 1})
    v_sum = g.nodes[c_sum].outputs[0]
    c_cos_off = g.new_call(CallOp.ADD, [v_cos], meta={"symbol": "cos(x1)-1", "literal": -1})
    g.nodes[c_cos_off].outputs  # output value registered
    _ = v_sum
    return g
