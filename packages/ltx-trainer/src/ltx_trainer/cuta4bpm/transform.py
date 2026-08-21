"""CUTA4BPM → BPMN flattening transformation (Sec. 4, Mendling strategy)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ltx_trainer.cuta4bpm.bpmn_graph import BpmnGraph
from ltx_trainer.cuta4bpm.metamodel import Block, CutaProcess, SimpleActivity


@dataclass
class SubgraphEndpoints:
    first: str
    last: str


def transform_activity(act: SimpleActivity, g: BpmnGraph, *, lane: str) -> SubgraphEndpoints:
    """SimpleActivity → BPMN task (+ lane), Fig. 6."""
    label = act.sentence()
    tid = g.add_node("task", label, role=act.role or lane, activity=act.to_dict())
    if act.role and not any(n.kind == "lane" and n.label == act.role for n in g.nodes):
        g.add_node("lane", act.role)
    return SubgraphEndpoints(tid, tid)


def transform_block(block: Block, g: BpmnGraph) -> SubgraphEndpoints:
    """Map one CUTA block to a BPMN subgraph with entry/exit (Fig. 4)."""
    if block.kind == "Sequence":
        return _transform_sequence(block, g)
    if block.kind == "Case":
        return _transform_case(block, g)
    if block.kind == "Parallel":
        return _transform_parallel(block, g, inclusive=False)
    if block.kind == "MultipleChoice":
        return _transform_parallel(block, g, inclusive=True)
    if block.kind == "Loop":
        return _transform_loop(block, g)
    raise ValueError(f"unknown block kind: {block.kind}")


def _transform_element(el: Any, g: BpmnGraph) -> SubgraphEndpoints:
    if isinstance(el, SimpleActivity):
        return transform_activity(el, g, lane=el.role)
    if isinstance(el, Block):
        return transform_block(el, g)
    raise TypeError(type(el))


def _transform_sequence(block: Block, g: BpmnGraph) -> SubgraphEndpoints:
    if not block.children:
        gid = g.add_node("exclusive_gateway", "empty-seq")
        return SubgraphEndpoints(gid, gid)
    first_ep: SubgraphEndpoints | None = None
    prev_last: str | None = None
    for child in block.children:
        ep = _transform_element(child, g)
        if first_ep is None:
            first_ep = ep
        if prev_last is not None:
            g.connect(prev_last, ep.first)
        prev_last = ep.last
    assert first_ep is not None and prev_last is not None
    return SubgraphEndpoints(first_ep.first, prev_last)


def _transform_case(block: Block, g: BpmnGraph) -> SubgraphEndpoints:
    fork = g.add_node("exclusive_gateway", "case-fork")
    join = g.add_node("exclusive_gateway", "case-join")
    conds = block.conditions or [f"branch{i}" for i in range(len(block.children))]
    for child, cond in zip(block.children, conds):
        ep = _transform_element(child, g)
        g.connect(fork, ep.first, condition=cond)
        g.connect(ep.last, join)
    return SubgraphEndpoints(fork, join)


def _transform_parallel(block: Block, g: BpmnGraph, *, inclusive: bool) -> SubgraphEndpoints:
    gw_kind = "inclusive_gateway" if inclusive else "parallel_gateway"
    fork = g.add_node(gw_kind, f"{block.kind}-fork")
    join = g.add_node(gw_kind, f"{block.kind}-join")
    for child in block.children:
        ep = _transform_element(child, g)
        g.connect(fork, ep.first)
        g.connect(ep.last, join)
    return SubgraphEndpoints(fork, join)


def _transform_loop(block: Block, g: BpmnGraph) -> SubgraphEndpoints:
    fork = g.add_node("exclusive_gateway", "loop-fork")
    join = g.add_node("exclusive_gateway", "loop-join")
    body = _transform_sequence(
        Block(kind="Sequence", children=list(block.children)),
        g,
    )
    cond = block.loop_condition or "while-true"
    if block.loop_position == "begin":
        g.connect(fork, body.first, condition=cond)
        g.connect(body.last, fork)
        g.connect(fork, join, condition="exit")
    else:
        g.connect(fork, body.first)
        g.connect(body.last, join, condition=cond)
        g.connect(join, fork)
    return SubgraphEndpoints(fork, join)


def transform_process(proc: CutaProcess) -> BpmnGraph:
    """CUTA4BPM workflow → BPMN process diagram (Fig. 5): pool, start, body, end."""
    g = BpmnGraph(pool=proc.company_pool)
    g.add_node("pool", proc.company_pool)
    start = g.add_node("start_event", "start")
    end = g.add_node("end_event", "end")
    body = _transform_element(proc.root, g)
    g.connect(start, body.first)
    g.connect(body.last, end)
    return g
