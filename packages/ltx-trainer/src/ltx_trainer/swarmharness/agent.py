"""Autonomous agent dispatch over SwarmHarness (Sec. 5.2, 6 — same MCP/credit as humans)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.swarmharness.config import SwarmHarnessConfig, SwarmNode
from ltx_trainer.swarmharness.identity import NodeIdentity
from ltx_trainer.swarmharness.orchestrator import LocalSwarm, TaskExecutor
from ltx_trainer.swarmharness.router import SwarmTask


@dataclass
class SubTaskSpec:
    task_id: str
    skill: str
    credit_pool: float


@dataclass
class SwarmAgent:
    """
    Agent with keypair + node identity: earns credits by serving, spends on subtasks.
    Indistinguishable from a human submitter at the protocol layer (Sec. 6).
    """

    identity: NodeIdentity
    swarm: LocalSwarm = field(default_factory=LocalSwarm)
    cfg: SwarmHarnessConfig = field(default_factory=SwarmHarnessConfig)

    def ensure_node_registered(self, skills: set[str], *, skip_pow: bool = True) -> None:
        from ltx_trainer.swarmharness.config import ResourceVector, SwarmNode

        if self.identity.node_id not in self.swarm.nodes:
            node = SwarmNode(
                node_id=self.identity.node_id,
                skills=skills,
                resources=ResourceVector(),
                public_key_hex=self.identity.public_key_hex,
            )
            self.swarm.register(node, identity=self.identity, skip_pow=skip_pow)

    def decompose_goal(self, goal: str) -> list[SubTaskSpec]:
        """Trivial planner: map keywords to skills (demo only)."""
        specs: list[SubTaskSpec] = []
        g = goal.lower()
        if "infer" in g or "gpu" in g:
            specs.append(SubTaskSpec("st-infer", "inference", 1.0))
        if "summar" in g or "text" in g:
            specs.append(SubTaskSpec("st-sum", "summarize", 0.5))
        if "embed" in g or "vector" in g:
            specs.append(SubTaskSpec("st-emb", "embed", 0.3))
        if not specs:
            specs.append(SubTaskSpec("st-default", "inference", 1.0))
        return specs

    def run_goal(
        self,
        goal: str,
        executor: TaskExecutor,
        *,
        serve_skills: set[str] | None = None,
    ) -> dict[str, Any]:
        serve_skills = serve_skills or {"inference", "summarize"}
        self.ensure_node_registered(serve_skills)
        self.swarm.ledger.ensure_account(self.identity.node_id, initial=self.cfg.genesis_credit)

        results: list[dict[str, Any]] = []
        for spec in self.decompose_goal(goal):
            task = SwarmTask(
                task_id=spec.task_id,
                skill=spec.skill,
                submitter_id=self.identity.node_id,
                credit_pool=spec.credit_pool,
            )
            try:
                out = self.swarm.run_task(task, executor, submitter_identity=self.identity)
                results.append({"task_id": spec.task_id, "ok": True, "route": out["route"]["selected_node"]})
            except Exception as exc:
                results.append({"task_id": spec.task_id, "ok": False, "error": str(exc)})

        return {
            "agent_id": self.identity.node_id,
            "goal": goal,
            "subtasks": results,
            "balance": self.swarm.ledger.balance(self.identity.node_id),
            "trust": self.swarm.trust.get(self.identity.node_id, 0.0),
        }


def demo_autonomous_agent(*, seed: int = 0) -> dict[str, Any]:
    """Smoke: agent decomposes goal and dispatches to local swarm."""
    import random

    from ltx_trainer.swarmharness.pipeline import demo_swarm_nodes

    swarm = LocalSwarm()
    swarm._rng = random.Random(seed)
    for n in demo_swarm_nodes():
        swarm.register(n, skip_pow=True)

    agent = SwarmAgent(identity=NodeIdentity.generate(), swarm=swarm)
    agent.swarm.ledger.ensure_account(agent.identity.node_id, initial=8.0)

    def _exec(worker: SwarmNode, _task: SwarmTask) -> float:
        return 0.7 + 0.2 * worker.trust

    return agent.run_goal("run gpu inference then summarize results", _exec, serve_skills={"inference", "summarize"})
