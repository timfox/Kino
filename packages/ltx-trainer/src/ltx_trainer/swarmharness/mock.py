"""SwarmHarness evaluation smoke (arXiv:2605.28764)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.swarmharness.config import SwarmHarnessConfig
from ltx_trainer.swarmharness.pipeline import evaluation_demo


def evaluation_smoke(cfg: SwarmHarnessConfig | None = None) -> dict[str, Any]:
    _ = cfg
    from ltx_trainer.swarmharness.agent import demo_autonomous_agent
    from ltx_trainer.swarmharness.orchestrator import demo_local_swarm

    demo = evaluation_demo(seed=0)
    local = demo_local_swarm(seed=0)
    agent = demo_autonomous_agent(seed=0)
    return {
        "paper": "arXiv:2605.28764",
        "selected_node": demo.get("selected_node"),
        "efficiency_error": demo["attribution"]["efficiency_error"],
        "registry_ads": demo["registry"]["advertisements"],
        "signed_receipt_ok": bool(local.get("signed_receipt")),
        "ledger_accounts": local["ledger_summary"]["accounts"],
        "agent_subtasks_ok": sum(1 for s in agent["subtasks"] if s.get("ok")),
    }
