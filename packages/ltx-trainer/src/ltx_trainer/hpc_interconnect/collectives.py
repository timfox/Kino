"""MPI-style collective cost models (custom ring AllGather, linear AlltoAll)."""

from __future__ import annotations

import math
from enum import Enum


class VictimCollective(str, Enum):
    ALLGATHER_RING = "allgather_ring"
    ALLTOALL_LINEAR = "alltoall_linear"


class AggressorCollective(str, Enum):
    ALLTOALL = "alltoall"
    INCAST = "incast"


def interleave_victim_aggressor(node_count: int) -> tuple[list[int], list[int]]:
    """Alternate nodes: even indices victims, odd aggressors (Sec. III-A)."""
    victims = [i for i in range(node_count) if i % 2 == 0]
    aggressors = [i for i in range(node_count) if i % 2 == 1]
    return victims, aggressors


def collective_cost_us(
    collective: VictimCollective | AggressorCollective,
    *,
    nodes: int,
    message_bytes: int,
    per_link_gbps: float = 100.0,
) -> float:
    """Synthetic communication time in microseconds (no MPI)."""
    n = max(2, nodes)
    b = max(8, message_bytes)
    bytes_per_sec = per_link_gbps * 1e9 / 8.0
    hop_time_s = b / bytes_per_sec

    if collective in (VictimCollective.ALLGATHER_RING,):
        steps = n - 1
        return steps * hop_time_s * 1e6 * 1.1
    if collective in (AggressorCollective.ALLTOALL, VictimCollective.ALLTOALL_LINEAR):
        steps = n - 1
        return steps * hop_time_s * 1e6 * n * 0.15
    if collective == AggressorCollective.INCAST:
        return (n // 2) * hop_time_s * 1e6 * 2.5
    return hop_time_s * 1e6


def message_bytes_from_label(label: str) -> int:
    mapping = {
        "512b": 512,
        "32kib": 32 * 1024,
        "2mib": 2 * 1024 * 1024,
        "16mib": 16 * 1024 * 1024,
    }
    return mapping.get(label.lower(), 512)


def log2_message_axis(min_bytes: int = 8, max_bytes: int = 16 * 1024 * 1024) -> list[int]:
    out: list[int] = []
    v = min_bytes
    while v <= max_bytes:
        out.append(v)
        v *= 2
    return out


def aggregate_runtime_ms(samples_us: list[float], *, drop_warmup: int = 100) -> float:
    """Mean of samples after warmup (Sec. III)."""
    if len(samples_us) <= drop_warmup:
        return float(sum(samples_us) / max(1, len(samples_us)) / 1000.0)
    tail = samples_us[drop_warmup:]
    return float(sum(tail) / len(tail) / 1000.0)


def simulate_iterations(
    base_us: float,
    *,
    iterations: int = 200,
    congestion_factor: float = 1.0,
    noise_fraction: float = 0.02,
) -> list[float]:
    """Generate per-iteration times with congestion slowdown."""
    import random

    rng = random.Random(42)
    out: list[float] = []
    for i in range(iterations):
        warm = 1.0 + 0.15 * math.exp(-i / 25.0) if i < 100 else 1.0
        jitter = 1.0 + noise_fraction * (rng.random() - 0.5)
        out.append(base_us * warm * congestion_factor * jitter)
    return out
