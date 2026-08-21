"""SwarmHarness configuration and node state (arXiv:2605.28764)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class RouterWeights:
    """Weights w1..w4 for utility function (eq. 1); must sum to 1."""

    capability: float = 0.35
    load: float = 0.25
    latency: float = 0.20
    trust: float = 0.20

    def __post_init__(self) -> None:
        total = self.capability + self.load + self.latency + self.trust
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"router weights must sum to 1, got {total}")


@dataclass
class ResourceVector:
    """Available resources on a node (paper rv)."""

    vram_gb: float = 0.0
    cpu_fraction: float = 1.0
    bandwidth_mbps: float = 100.0


@dataclass
class SwarmHarnessConfig:
    """Protocol parameters for SwarmRegistry, SwarmRouter, and SwarmCredit."""

    # Registry (Sec. 3.2)
    refresh_interval_s: float = 60.0
    eviction_multiplier: float = 3.0  # evict after 3 * T_refresh without refresh

    # Router (Sec. 3.3)
    router_weights: RouterWeights = field(default_factory=RouterWeights)
    latency_ceiling_ms: float = 500.0
    top_k_redundant: int = 1

    # SwarmCredit (Sec. 4)
    shapley_samples: int = 100
    trust_learning_rate: float = 0.1
    trust_decay_beta: float = 0.05
    trust_decay_period_s: float = 86400.0  # T0 = 24h
    default_credit_pool: float = 1.0

    # Genesis / cold-start (Sec. 4.5)
    genesis_credit: float = 10.0
    genesis_task_grant: float = 1.0
    min_vram_gb_gossip: float = 0.0

    # Sybil registration (Sec. 5.4) — stub difficulty bits
    registration_pow_bits: int = 12


@dataclass
class SwarmNode:
    """SwarmNode v = (Sv, rv, cv, τv) augmented with load and latency (Sec. 3.1)."""

    node_id: str
    skills: set[str] = field(default_factory=set)
    resources: ResourceVector = field(default_factory=ResourceVector)
    credit: float = 0.0
    trust: float = 0.0
    load_fraction: float = 0.0
    latency_ms: float = 0.0
    genesis_locked: bool = True
    genesis_unlocked: bool = False
    public_key_hex: str = ""

    def capability_match(self, skill: str) -> bool:
        return skill in self.skills


def normalise_weights(weights: Sequence[float]) -> tuple[float, ...]:
    s = float(sum(weights))
    if s <= 0:
        raise ValueError("weights must be positive")
    return tuple(w / s for w in weights)
