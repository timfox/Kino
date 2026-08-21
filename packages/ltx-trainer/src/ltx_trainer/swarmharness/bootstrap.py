"""Bootstrap discovery config: mDNS, DNS seeds, explicit peers (Sec. 3.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BootstrapConfig:
    """Ordered bootstrap mechanisms (first match wins in production overlay)."""

    mdns_enabled: bool = True
    dns_seeds: list[str] = field(default_factory=lambda: ["seed1.swarmharness.io", "seed2.swarmharness.io"])
    explicit_peers: list[str] = field(default_factory=list)

    @classmethod
    def local_private(cls, peers: list[str]) -> BootstrapConfig:
        """Phase-1 private swarm: peer list only (Sec. 5.3)."""
        return cls(mdns_enabled=False, dns_seeds=[], explicit_peers=list(peers))

    def plan(self) -> list[dict[str, str]]:
        steps: list[dict[str, str]] = []
        if self.mdns_enabled:
            steps.append({"step": "mdns", "action": "multicast discover on LAN"})
        if self.dns_seeds:
            steps.append({"step": "dns", "action": f"resolve {len(self.dns_seeds)} seed(s)"})
        if self.explicit_peers:
            steps.append({"step": "peers", "action": f"connect {len(self.explicit_peers)} explicit peer(s)"})
        if not steps:
            steps.append({"step": "none", "action": "isolated swarm — add peers or enable mDNS"})
        return steps


def parse_peer_list(raw: str) -> list[str]:
    """Comma- or newline-separated host:port peers."""
    parts: list[str] = []
    for chunk in raw.replace("\n", ",").split(","):
        p = chunk.strip()
        if p:
            parts.append(p)
    return parts


def bootstrap_card(cfg: BootstrapConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BootstrapConfig()
    return {
        "mdns": cfg.mdns_enabled,
        "dns_seeds": list(cfg.dns_seeds),
        "explicit_peers": list(cfg.explicit_peers),
        "plan": cfg.plan(),
    }
