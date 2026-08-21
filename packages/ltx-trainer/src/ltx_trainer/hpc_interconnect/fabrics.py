"""Fabric profiles for congestion simulation (Sec. II–III)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SystemName(str, Enum):
    LEONARDO = "leonardo"
    CRESCO8 = "cresco8"
    LUMI = "lumi"
    HAICGU = "haicgu"
    NANJING = "nanjing"


class AggressorPattern(str, Enum):
    ALLTOALL = "alltoall"
    INCAST = "incast"


@dataclass(frozen=True)
class FabricProfile:
    """Analytic congestion model tuned to paper Fig. 4–6 anchors."""

    name: SystemName
    partition_nodes: int
    interconnect: str
    topology: str
    # Steady-state resilience in [0, 1] at reference scale (64 nodes, ~2% partition)
    alltoall_resilience: float
    incast_resilience: float
    scale_penalty: float  # larger → worse as node count grows
    partition_fraction_penalty: float  # CRESCO8 uses 33% at 256 nodes
    bursty_recovery: float  # 0–1, higher recovers between bursts
    nslb_enabled: bool = True

    def partition_fraction(self, allocated_nodes: int) -> float:
        return min(1.0, allocated_nodes / max(1, self.partition_nodes))


_PROFILES: dict[SystemName, FabricProfile] = {
    SystemName.LEONARDO: FabricProfile(
        name=SystemName.LEONARDO,
        partition_nodes=3456,
        interconnect="HDR InfiniBand",
        topology="Dragonfly+",
        alltoall_resilience=0.97,
        incast_resilience=0.55,
        scale_penalty=0.35,
        partition_fraction_penalty=0.15,
        bursty_recovery=0.55,
    ),
    SystemName.CRESCO8: FabricProfile(
        name=SystemName.CRESCO8,
        partition_nodes=760,
        interconnect="NDR InfiniBand",
        topology="1.67:1 Fat-Tree",
        alltoall_resilience=0.72,
        incast_resilience=0.78,
        scale_penalty=0.55,
        partition_fraction_penalty=0.45,
        bursty_recovery=0.50,
    ),
    SystemName.LUMI: FabricProfile(
        name=SystemName.LUMI,
        partition_nodes=2978,
        interconnect="Cray Slingshot",
        topology="Dragonfly",
        alltoall_resilience=0.99,
        incast_resilience=0.98,
        scale_penalty=0.08,
        partition_fraction_penalty=0.05,
        bursty_recovery=0.92,
    ),
    SystemName.HAICGU: FabricProfile(
        name=SystemName.HAICGU,
        partition_nodes=10,
        interconnect="EDR / RoCE CE8850",
        topology="single switch",
        alltoall_resilience=0.55,
        incast_resilience=0.70,
        scale_penalty=0.20,
        partition_fraction_penalty=0.10,
        bursty_recovery=0.40,
        nslb_enabled=False,
    ),
    SystemName.NANJING: FabricProfile(
        name=SystemName.NANJING,
        partition_nodes=8,
        interconnect="RoCE NSLB CE9855",
        topology="2-leaf 2-spine",
        alltoall_resilience=0.98,
        incast_resilience=0.90,
        scale_penalty=0.12,
        partition_fraction_penalty=0.08,
        bursty_recovery=0.75,
        nslb_enabled=True,
    ),
}


def get_system(name: SystemName | str) -> FabricProfile:
    return _PROFILES[SystemName(name)]
