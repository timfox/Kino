"""Paper positioning, security surface, and open challenges (arXiv:2605.28764)."""

from __future__ import annotations

from typing import Any


def positioning_table() -> list[dict[str, str]]:
    """Sec. 2.6 — three-way distinction vs prior systems."""
    return [
        {
            "system": "BOINC / Petals",
            "skill_first_mcp": "no",
            "decentralised_dht": "partial (Petals P2P)",
            "shapley_credit_no_chain": "no",
        },
        {
            "system": "Golem / BrokerChain",
            "skill_first_mcp": "no",
            "decentralised_dht": "yes",
            "shapley_credit_no_chain": "no (on-chain)",
        },
        {
            "system": "SwarmHarness",
            "skill_first_mcp": "yes (HarnessAPI)",
            "decentralised_dht": "yes",
            "shapley_credit_no_chain": "yes",
        },
    ]


def security_surface() -> list[dict[str, str]]:
    """Sec. 5.4 — mitigations implemented or stubbed in this repo."""
    return [
        {"threat": "Sybil identities", "mitigation": "Registration PoW + τv=0 for new nodes"},
        {"threat": "Credit inflation / collusion", "mitigation": "Submitter countersigned CreditReceipt"},
        {"threat": "Malicious router", "mitigation": "Router stateless; attribution signed by submitter"},
        {"threat": "Task payload leakage", "mitigation": "Router/registry store no payloads (design)"},
        {"threat": "Skill escape", "mitigation": "HarnessAPI subprocess sandbox (external)"},
    ]


def open_challenges() -> list[dict[str, str]]:
    """Sec. 5.5 — research directions."""
    return [
        {"topic": "Credit valuation", "note": "Ordinal credits vs external exchange rate / auctions"},
        {"topic": "Heterogeneous quality signals", "note": "Subjective tasks need delayed human ratings"},
        {"topic": "Byzantine fault tolerance", "note": "ZK proofs of correct execution"},
        {"topic": "Dynamic skill pricing", "note": "Per-skill market clearing without central auctioneer"},
    ]


def related_work_anchors() -> dict[str, str]:
    """Key citations mapped to SwarmHarness design choices."""
    return {
        "HarnessAPI": "arXiv:2026 — skill folder → HTTP + MCP",
        "Kademlia": "IPTPS 2002 — DHT key = SHA256(skill)",
        "FedToken / VerFedSV": "Shapley attribution for incentives",
        "Lattica": "NAT traversal for P2P inference",
        "Vellinger et al.": "Digital pheromones ↔ trust/credit signals",
    }


def deployment_phases() -> list[dict[str, str]]:
    """Sec. 5.1 deployment path."""
    return [
        {"phase": "1", "months": "0-6", "action": "pip addon on HarnessAPI nodes; local swarms via mDNS"},
        {"phase": "2", "months": "6-24", "action": "Federated DNS seeds; credit economy self-regulation"},
        {"phase": "3", "months": "2y+", "action": "Full bootstrap independence; autonomous agent mesh"},
    ]
