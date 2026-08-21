"""Paper evaluation scenario demos."""

from __future__ import annotations

from ltx_trainer.cbp_assist.workflow import run_workflow


def demo_knowledge_query() -> dict:
    state = run_workflow("Which capabilities can drill a workpiece?")
    return {
        "intent": state.intent,
        "reply": state.natural_language_reply,
        "trace": state.trace,
        "ok": "drill" in state.natural_language_reply.lower(),
    }


def demo_sat_planning() -> dict:
    state = run_workflow("Plan to drill 7 mm depth at station 3.")
    return {
        "satisfiable": state.planning.satisfiable if state.planning else False,
        "plan_steps": len(state.planning.plan) if state.planning else 0,
        "reply": state.natural_language_reply,
        "ok": bool(state.planning and state.planning.satisfiable),
    }


def demo_unsat_planning() -> dict:
    state = run_workflow(
        "Plan drilling with depth 2 mm at station 15.",
        auto_approve_hitl=True,
        apply_adaptation=False,
    )
    return {
        "satisfiable": state.planning.satisfiable if state.planning else True,
        "conflicts": state.planning.unsat_core if state.planning else [],
        "proposal": state.adaptation_proposal,
        "ok": bool(state.planning and not state.planning.satisfiable and state.adaptation_proposal),
    }


def demo_adaptive_planning() -> dict:
    state = run_workflow(
        "Plan drilling with depth 2 mm at station 15.",
        auto_approve_hitl=True,
        apply_adaptation=True,
    )
    return {
        "satisfiable_after_adaptation": state.planning.satisfiable if state.planning else False,
        "reply": state.natural_language_reply,
        "trace_tail": state.trace[-4:],
        "ok": bool(state.planning and state.planning.satisfiable),
    }


def demo_runtime_failure() -> dict:
    state = run_workflow("The conveyor is defective.")
    return {
        "conveyor_removed": "cap_conveyor" not in state.provided,
        "satisfiable": state.planning.satisfiable if state.planning else False,
        "ok": "cap_conveyor" not in state.provided and bool(state.planning and state.planning.satisfiable),
    }
