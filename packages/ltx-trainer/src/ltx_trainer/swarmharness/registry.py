"""SwarmRegistry: Kademlia-style skill index with TTL advertisements (Sec. 3.2)."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.swarmharness.config import SwarmHarnessConfig, SwarmNode
from ltx_trainer.swarmharness.kademlia import closest_nodes


def skill_dht_key(skill_name: str) -> str:
    """SHA256(skill_name) as hex — Kademlia key (Sec. 3.2)."""
    return hashlib.sha256(skill_name.encode("utf-8")).hexdigest()


@dataclass
class CapabilityAdvertisement:
    """⟨node_id, Sv, rv, τv⟩ published to the DHT (Sec. 3.1–3.2)."""

    node_id: str
    skills: frozenset[str]
    vram_gb: float
    cpu_fraction: float
    trust: float
    public_key_hex: str
    advertised_at: float = field(default_factory=time.time)

    def to_node_snapshot(self, *, load_fraction: float = 0.0, latency_ms: float = 0.0) -> SwarmNode:
        from ltx_trainer.swarmharness.config import ResourceVector

        return SwarmNode(
            node_id=self.node_id,
            skills=set(self.skills),
            resources=ResourceVector(
                vram_gb=self.vram_gb,
                cpu_fraction=self.cpu_fraction,
            ),
            trust=self.trust,
            load_fraction=load_fraction,
            latency_ms=latency_ms,
            public_key_hex=self.public_key_hex,
        )


@dataclass
class SwarmRegistry:
    """In-memory DHT stub: skill key → TTL-bounded node advertisements."""

    cfg: SwarmHarnessConfig = field(default_factory=SwarmHarnessConfig)
    _by_skill: dict[str, list[CapabilityAdvertisement]] = field(default_factory=dict)
    _gossip_vram: dict[str, list[str]] = field(default_factory=dict)  # node_id by min vram bucket

    def publish(self, node: SwarmNode) -> None:
        """Refresh advertisements for all skills on node."""
        now = time.time()
        ad = CapabilityAdvertisement(
            node_id=node.node_id,
            skills=frozenset(node.skills),
            vram_gb=node.resources.vram_gb,
            cpu_fraction=node.resources.cpu_fraction,
            trust=node.trust,
            public_key_hex=node.public_key_hex,
            advertised_at=now,
        )
        for skill in node.skills:
            key = skill_dht_key(skill)
            entries = [e for e in self._by_skill.get(key, []) if e.node_id != node.node_id]
            entries.append(ad)
            self._by_skill[key] = entries
        self._gossip_upsert(node.node_id, node.resources.vram_gb)

    def _gossip_upsert(self, node_id: str, vram_gb: float) -> None:
        for ids in self._gossip_vram.values():
            if node_id in ids:
                ids.remove(node_id)
        bucket = int(vram_gb)
        self._gossip_vram.setdefault(str(bucket), []).append(node_id)

    def evict_stale(self, now: float | None = None) -> int:
        """Remove entries older than eviction_multiplier * refresh_interval."""
        now = now if now is not None else time.time()
        ttl = self.cfg.refresh_interval_s * self.cfg.eviction_multiplier
        removed = 0
        for key, entries in list(self._by_skill.items()):
            kept = [e for e in entries if (now - e.advertised_at) <= ttl]
            removed += len(entries) - len(kept)
            if kept:
                self._by_skill[key] = kept
            else:
                del self._by_skill[key]
        return removed

    def lookup_skill(self, skill: str) -> list[CapabilityAdvertisement]:
        self.evict_stale()
        return list(self._by_skill.get(skill_dht_key(skill), []))

    def lookup_skill_kademlia(self, skill: str, *, k: int = 8) -> list[CapabilityAdvertisement]:
        """Kademlia-closest node ids, then filter to skill advertisements."""
        self.evict_stale()
        ads = self.lookup_skill(skill)
        if not ads:
            return []
        ordered_ids = closest_nodes(skill, [a.node_id for a in ads], k=k)
        by_id = {a.node_id: a for a in ads}
        return [by_id[nid] for nid in ordered_ids if nid in by_id]

    def lookup_vram_at_least(self, min_vram_gb: float) -> list[str]:
        """Secondary gossip index: nodes with VRAM ≥ threshold (Sec. 3.2)."""
        out: list[str] = []
        min_bucket = int(min_vram_gb)
        for bucket, ids in self._gossip_vram.items():
            if int(bucket) >= min_bucket:
                out.extend(ids)
        return sorted(set(out))

    def stats(self) -> dict[str, Any]:
        return {
            "skill_keys": len(self._by_skill),
            "advertisements": sum(len(v) for v in self._by_skill.values()),
            "gossip_buckets": len(self._gossip_vram),
        }
