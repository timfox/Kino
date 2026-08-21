"""SwarmHarness framework card, decentralisation roadmap, evaluation demo (arXiv:2605.28764)."""

from __future__ import annotations

import random
from typing import Any

from ltx_trainer.swarmharness.config import ResourceVector, SwarmHarnessConfig, SwarmNode
from ltx_trainer.swarmharness.credit import (
    GenesisState,
    attribution_summary,
    grant_node_genesis,
    grant_submitter_genesis_task,
    leave_one_out_quality_proxy,
    shapley_standard_error,
    swarm_credit_attribution,
    trust_decay,
    unlock_genesis_after_serve,
)
from ltx_trainer.swarmharness.registry import SwarmRegistry, skill_dht_key
from ltx_trainer.swarmharness.router import SwarmTask, route_task, routing_report, utility_score
from ltx_trainer.swarmharness.tables import (
    deployment_phases,
    open_challenges,
    positioning_table,
    related_work_anchors,
    security_surface,
)


def framework_card(cfg: SwarmHarnessConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SwarmHarnessConfig()
    w = cfg.router_weights
    return {
        "name": "SwarmHarness",
        "paper": "arXiv:2605.28764",
        "authors": "Edwin Jose (Western Michigan University)",
        "tagline": "Skill-based task routing via decentralised incentive-aligned HarnessAPI swarms",
        "components": [
            "SwarmRegistry (Kademlia DHT + gossip VRAM index)",
            "SwarmRouter (utility scoring, eq. 1)",
            "SwarmCredit (Shapley MC attribution, Algorithm 1)",
        ],
        "harnessapi": "Layered on HarnessAPI skill nodes; backward compatible skill folders",
        "decentralisation": decentralisation_phases(),
        "config": {
            "refresh_interval_s": cfg.refresh_interval_s,
            "shapley_samples": cfg.shapley_samples,
            "genesis_credit": cfg.genesis_credit,
            "router_weights": {
                "capability": w.capability,
                "load": w.load,
                "latency": w.latency,
                "trust": w.trust,
            },
        },
    }


def decentralisation_phases() -> list[dict[str, str]]:
    """Sec. 3.5 — operational vs administrative decentralisation roadmap."""
    return [
        {
            "phase": "1",
            "label": "alpha",
            "operational": "full",
            "administrative": "single reference seed",
        },
        {
            "phase": "2",
            "label": "federated",
            "operational": "full",
            "administrative": "community DNS seed list",
        },
        {
            "phase": "3",
            "label": "fully decentralised",
            "operational": "full",
            "administrative": "mDNS + peer lists; DNS optional",
        },
    ]


def bootstrap_mechanisms() -> list[dict[str, str]]:
    return [
        {"order": "1", "mechanism": "mDNS", "use": "same-LAN zero-config"},
        {"order": "2", "mechanism": "DNS seed list", "use": "Bitcoin-style entry points"},
        {"order": "3", "mechanism": "explicit peer list", "use": "private / air-gapped swarms"},
    ]


def demo_swarm_nodes() -> list[SwarmNode]:
    """Three-node toy swarm matching Figure 1."""
    return [
        SwarmNode(
            node_id="node-a",
            skills={"inference", "summarize"},
            resources=ResourceVector(vram_gb=24.0, cpu_fraction=0.4),
            trust=0.85,
            load_fraction=0.2,
            latency_ms=12.0,
            credit=5.0,
            genesis_locked=False,
            genesis_unlocked=True,
        ),
        SwarmNode(
            node_id="node-b",
            skills={"inference", "code_analysis", "embed"},
            resources=ResourceVector(vram_gb=8.0, cpu_fraction=0.7),
            trust=0.55,
            load_fraction=0.6,
            latency_ms=45.0,
            credit=2.0,
        ),
        SwarmNode(
            node_id="node-c",
            skills={"summarize"},
            resources=ResourceVector(vram_gb=0.0, cpu_fraction=0.15),
            trust=0.40,
            load_fraction=0.1,
            latency_ms=80.0,
            credit=1.0,
        ),
    ]


def evaluation_demo(*, seed: int = 42) -> dict[str, Any]:
    """
    End-to-end smoke: registry publish → route → Shapley credit → genesis unlock.
    """
    cfg = SwarmHarnessConfig(shapley_samples=50)
    rng = random.Random(seed)
    nodes = demo_swarm_nodes()
    registry = SwarmRegistry(cfg=cfg)
    for n in nodes:
        registry.publish(n)

    task = SwarmTask(task_id="demo-1", skill="inference", submitter_id="user-1", credit_pool=2.0)
    selected = route_task(nodes, task, cfg=cfg, rng=rng)
    report = routing_report(nodes, task, cfg=cfg)

    contributors = ["node-a", "node-b"] if selected else []
    weights = {"node-a": 0.55, "node-b": 0.35}
    quality_fn = lambda s: leave_one_out_quality_proxy(s, node_weights=weights, baseline=0.05)
    trust = {n.node_id: n.trust for n in nodes}
    attr = swarm_credit_attribution(
        contributors,
        quality_fn,
        task.credit_pool or cfg.default_credit_pool,
        trust,
        submitter_balance=10.0,
        cfg=cfg,
        rng=rng,
    )

    genesis = grant_node_genesis(GenesisState(), cfg=cfg)
    genesis = unlock_genesis_after_serve(genesis)
    submitter = grant_submitter_genesis_task(GenesisState(), cfg=cfg)

    idle_trust = trust_decay(0.9, dt_seconds=cfg.trust_decay_period_s, beta=cfg.trust_decay_beta, period_seconds=cfg.trust_decay_period_s)

    return {
        "registry": registry.stats(),
        "skill_dht_key_inference": skill_dht_key("inference")[:16] + "...",
        "routing": report,
        "selected_node": selected.node_id if selected else None,
        "attribution": attribution_summary(attr),
        "shapley_stderr_bound": round(shapley_standard_error(cfg.shapley_samples), 4),
        "genesis_node_unlocked": not genesis.locked and genesis.served_proof,
        "submitter_free_task": submitter.free_task_used,
        "trust_after_24h_idle": round(idle_trust, 4),
        "utility_winner_score": round(utility_score(selected, task, cfg=cfg), 4) if selected else None,
        "elapsed_ms": 0,
    }


def knowledge_card() -> dict[str, Any]:
    return {
        "arxiv": "2605.28764",
        "author": "Edwin Jose (Western Michigan University)",
        "components": ["SwarmRegistry", "SwarmRouter", "SwarmCredit"],
        "positioning": positioning_table(),
        "security": security_surface(),
        "open_challenges": open_challenges(),
        "related_work": related_work_anchors(),
        "deployment": deployment_phases(),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "pheromones": table_pheromone_feedback(),
        "positioning": positioning_table(),
        "security": security_surface(),
        "deployment": deployment_phases(),
    }


def table_pheromone_feedback() -> list[dict[str, str]]:
    """Sec. 4.7 — credit/trust as digital pheromones."""
    return [
        {"signal": "high trust + credit", "effect": "more routing traffic (positive feedback)"},
        {"signal": "high load ℓv", "effect": "utility penalty redirects traffic (negative feedback)"},
        {"signal": "poor quality on skill", "effect": "trust decay → specialisation away from skill"},
    ]
