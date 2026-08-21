"""Local swarm orchestrator: registry + router + signed SwarmCredit (Sec. 3, 5.1)."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any
from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.swarmharness.config import SwarmHarnessConfig, SwarmNode
from ltx_trainer.swarmharness.credit import (
    GenesisState,
    grant_node_genesis,
    grant_submitter_genesis_task,
    leave_one_out_quality_proxy,
    swarm_credit_attribution,
    trust_decay,
    unlock_genesis_after_serve,
)
from ltx_trainer.swarmharness.identity import (
    CreditReceipt,
    NodeIdentity,
    mine_registration_pow,
    sign_credit_receipt,
    verify_registration_pow,
)
from ltx_trainer.swarmharness.ledger import LocalCreditLedger
from ltx_trainer.swarmharness.registry import SwarmRegistry
from ltx_trainer.swarmharness.router import SwarmTask, route_task, route_top_k, routing_report


TaskExecutor = Callable[[SwarmNode, SwarmTask], float]


@dataclass
class RegisteredNode:
    node: SwarmNode
    identity: NodeIdentity | None = None
    registration_nonce: int | None = None
    genesis: GenesisState = field(default_factory=GenesisState)
    last_task_at: float = 0.0


@dataclass
class LocalSwarm:
    """
    In-process swarm for lab / private deployments (Phase 1, Sec. 5.3).

    Wires SwarmRegistry, SwarmRouter, and signed SwarmCredit without network I/O.
    """

    cfg: SwarmHarnessConfig = field(default_factory=SwarmHarnessConfig)
    registry: SwarmRegistry = field(init=False)
    ledger: LocalCreditLedger = field(default_factory=LocalCreditLedger)
    nodes: dict[str, RegisteredNode] = field(default_factory=dict)
    trust: dict[str, float] = field(default_factory=dict)
    _rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        self.registry = SwarmRegistry(cfg=self.cfg)

    def register(
        self,
        node: SwarmNode,
        *,
        identity: NodeIdentity | None = None,
        pow_nonce: int | None = None,
        skip_pow: bool = False,
    ) -> RegisteredNode:
        ident = identity
        if ident is not None:
            node = SwarmNode(
                node_id=ident.node_id,
                skills=set(node.skills),
                resources=node.resources,
                credit=node.credit,
                trust=node.trust,
                load_fraction=node.load_fraction,
                latency_ms=node.latency_ms,
                genesis_locked=node.genesis_locked,
                genesis_unlocked=node.genesis_unlocked,
                public_key_hex=ident.public_key_hex,
            )
        nonce = pow_nonce
        if not skip_pow and self.cfg.registration_pow_bits > 0:
            if nonce is None:
                nonce = mine_registration_pow(node.node_id, self.cfg.registration_pow_bits)
            if not verify_registration_pow(node.node_id, nonce, self.cfg.registration_pow_bits):
                raise ValueError("registration PoW verification failed")
        genesis = grant_node_genesis(GenesisState(credit=node.credit), cfg=self.cfg)
        self.ledger.ensure_account(node.node_id, initial=node.credit)
        self.trust.setdefault(node.node_id, node.trust)
        rec = RegisteredNode(node=node, identity=ident, registration_nonce=nonce, genesis=genesis)
        self.nodes[node.node_id] = rec
        self.registry.publish(node)
        return rec

    def apply_idle_trust_decay(self, now: float | None = None) -> dict[str, float]:
        """Eq. (2) — decay τ for nodes idle since last_task_at."""
        now = now if now is not None else time.time()
        updated: dict[str, float] = {}
        for nid, reg in self.nodes.items():
            dt = now - (reg.last_task_at or 0.0)
            if dt <= 0:
                continue
            tau = self.trust.get(nid, reg.node.trust)
            new_tau = trust_decay(
                tau,
                dt,
                beta=self.cfg.trust_decay_beta,
                period_seconds=self.cfg.trust_decay_period_s,
            )
            self.trust[nid] = new_tau
            reg.node.trust = new_tau
            updated[nid] = new_tau
        return updated

    def candidates_for(self, skill: str, *, min_vram_gb: float = 0.0) -> list[SwarmNode]:
        ads = self.registry.lookup_skill(skill)
        out: list[SwarmNode] = []
        seen: set[str] = set()
        for ad in ads:
            if ad.node_id in seen:
                continue
            seen.add(ad.node_id)
            snap = ad.to_node_snapshot()
            if min_vram_gb > 0 and snap.resources.vram_gb < min_vram_gb:
                continue
            reg = self.nodes.get(ad.node_id)
            if reg:
                snap.trust = self.trust.get(ad.node_id, snap.trust)
                snap.load_fraction = reg.node.load_fraction
                snap.latency_ms = reg.node.latency_ms
            out.append(snap)
        return out

    def submit_task(
        self,
        task: SwarmTask,
        *,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Route task and return routing report (execution via ``run_task``)."""
        pool = task.credit_pool if task.credit_pool is not None else self.cfg.default_credit_pool
        self.ledger.ensure_account(task.submitter_id)
        if self.ledger.balance(task.submitter_id) < pool:
            gs0 = GenesisState(credit=self.ledger.balance(task.submitter_id))
            gs1 = grant_submitter_genesis_task(gs0, cfg=self.cfg)
            if gs1.free_task_used and not gs0.free_task_used:
                self.ledger.balances[task.submitter_id] = gs1.credit
            if self.ledger.balance(task.submitter_id) < pool:
                raise ValueError(f"insufficient credit for submitter {task.submitter_id}")

        candidates = self.candidates_for(task.skill)
        if top_k and top_k > 1:
            selected = route_top_k(candidates, task, k=top_k, cfg=self.cfg)
            report = routing_report(candidates, task, cfg=self.cfg)
            return {
                "task_id": task.task_id,
                "credit_pool": pool,
                "routing": report,
                "selected_nodes": [n.node_id for n in selected],
            }
        winner = route_task(candidates, task, cfg=self.cfg, rng=self._rng)
        report = routing_report(candidates, task, cfg=self.cfg)
        return {
            "task_id": task.task_id,
            "credit_pool": pool,
            "routing": report,
            "selected_node": winner.node_id if winner else None,
        }

    def run_task(
        self,
        task: SwarmTask,
        executor: TaskExecutor,
        *,
        submitter_identity: NodeIdentity,
        node_weights: dict[str, float] | None = None,
        top_k: int = 1,
    ) -> dict[str, Any]:
        """
        Route, execute on worker(s), attribute Shapley credit, countersign, apply ledger.
        """
        route_info = self.submit_task(task, top_k=top_k)
        pool = float(route_info["credit_pool"])
        candidates = self.candidates_for(task.skill)

        if top_k > 1:
            workers = route_top_k(candidates, task, k=top_k, cfg=self.cfg)
        else:
            w = route_task(candidates, task, cfg=self.cfg, rng=self._rng)
            workers = [w] if w else []

        if not workers:
            raise RuntimeError(f"no node available for skill {task.skill!r}")

        qualities: dict[str, float] = {}
        for worker in workers:
            q = max(0.0, min(1.0, float(executor(worker, task))))
            qualities[worker.node_id] = q
            reg = self.nodes.get(worker.node_id)
            if reg:
                reg.last_task_at = time.time()
                if reg.genesis.locked:
                    reg.genesis = unlock_genesis_after_serve(reg.genesis)

        contributors = list(qualities.keys())
        weights = node_weights or {n: qualities[n] for n in contributors}
        quality_fn = lambda s: leave_one_out_quality_proxy(  # noqa: E731
            s, node_weights=weights, baseline=0.02
        )
        trust = {nid: self.trust.get(nid, 0.0) for nid in self.nodes}
        submitter_balance = self.ledger.balance(task.submitter_id)
        attr = swarm_credit_attribution(
            contributors,
            quality_fn,
            pool,
            trust,
            submitter_balance,
            cfg=self.cfg,
            rng=self._rng,
        )
        for nid, tau in attr.trust.items():
            self.trust[nid] = tau
            if nid in self.nodes:
                self.nodes[nid].node.trust = tau

        mean_q = sum(qualities.values()) / len(qualities)
        receipt = CreditReceipt.build(
            task_id=task.task_id,
            submitter_id=task.submitter_id,
            contributors=contributors,
            deltas=attr.deltas,
            credit_pool=pool,
            quality=mean_q,
        )
        signed = sign_credit_receipt(submitter_identity, receipt)
        ledger_apply = self.ledger.apply_signed_attribution(signed)

        return {
            "route": route_info,
            "qualities": qualities,
            "attribution": {
                "deltas": attr.deltas,
                "shapley_raw": attr.shapley_raw,
                "submitter_balance": self.ledger.balance(task.submitter_id),
            },
            "signed_receipt": signed,
            "ledger": ledger_apply,
            "ledger_summary": self.ledger.summary(),
        }


def demo_local_swarm(*, seed: int = 0) -> dict[str, Any]:
    """Run a two-node local swarm with stub executors."""
    from ltx_trainer.swarmharness.pipeline import demo_swarm_nodes

    cfg = SwarmHarnessConfig(registration_pow_bits=8, shapley_samples=40)
    swarm = LocalSwarm(cfg=cfg)
    swarm._rng = random.Random(seed)

    submitter = NodeIdentity.generate()
    swarm.ledger.ensure_account(submitter.node_id, initial=5.0)

    for n in demo_swarm_nodes():
        swarm.register(n, identity=None, skip_pow=True)

    task = SwarmTask(
        task_id="local-1",
        skill="inference",
        submitter_id=submitter.node_id,
        credit_pool=1.5,
    )

    def _exec(worker: SwarmNode, _task: SwarmTask) -> float:
        base = 0.5 + 0.4 * worker.trust - 0.2 * worker.load_fraction
        return max(0.0, min(1.0, base))

    return swarm.run_task(task, _exec, submitter_identity=submitter)
