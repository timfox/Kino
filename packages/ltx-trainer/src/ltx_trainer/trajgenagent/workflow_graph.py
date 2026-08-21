"""LangGraph-style worker workflow card (Sec. III-D, Fig. 2)."""

from __future__ import annotations

from ltx_trainer.trajgenagent.config import TrajGenAgentConfig
from ltx_trainer.trajgenagent.trajectory import Visit

WORKFLOW_NODES = ("location", "travel", "duration", "verify")
WORKFLOW_EDGES = (
    ("location", "travel"),
    ("travel", "duration"),
    ("duration", "verify"),
    ("verify", "location"),
    ("verify", "end"),
)


def verify_visit_bounds(visit: Visit, *, cfg: TrajGenAgentConfig | None = None) -> tuple[bool, str]:
    cfg = cfg or TrajGenAgentConfig()
    minutes = visit.duration_minutes
    if minutes < cfg.duration_min_minutes or minutes > cfg.duration_max_minutes:
        return False, "duration_bounds"
    if visit.end <= visit.start:
        return False, "temporal_order"
    return True, "ok"


def workflow_graph_card() -> dict[str, object]:
    return {
        "pattern": "deterministic LangGraph loop",
        "nodes": list(WORKFLOW_NODES),
        "edges": [{"from": a, "to": b} for a, b in WORKFLOW_EDGES],
        "visit_loop": "for each activity: location → travel → duration → verify",
        "tool_order_fixed": True,
        "freeform_tool_calling": False,
        "verifier_hooks": ["poi_schema", "kinematics_bounds", "duration_bounds", "temporal_order"],
    }

