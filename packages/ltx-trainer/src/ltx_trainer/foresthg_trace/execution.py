"""Traceable deterministic execution: S_{t+1}, r_t = f_{o_t}(S_t, u_t)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.foresthg_trace.hypergraph import SceneHypergraph
from ltx_trainer.foresthg_trace.operators import OperatorCall, ReasoningContext, StepResult, execute_operator


@dataclass
class ExecutionTrace:
    answer: str
    tool_trace: list[str]
    intermediate_states: list[dict[str, Any]]
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "final_answer": self.answer,
            "tool_trace": self.tool_trace,
            "intermediate_states": self.intermediate_states,
            "evidence": self.evidence,
        }


def run_program(
    graph: SceneHypergraph,
    program: list[OperatorCall],
    *,
    evidence: dict[str, Any] | None = None,
) -> ExecutionTrace:
    ctx = ReasoningContext()
    tool_trace: list[str] = []
    states: list[dict[str, Any]] = []
    answer = ""
    for call in program:
        step: StepResult = execute_operator(graph, ctx, call)
        ctx = step.context
        tool_trace.append(call.op)
        states.append({"op": call.op, "args": call.args, "result": step.payload})
        if call.op == "answer":
            answer = str(step.payload.get("answer", ""))
    return ExecutionTrace(
        answer=answer,
        tool_trace=tool_trace,
        intermediate_states=states,
        evidence=evidence or {},
    )
