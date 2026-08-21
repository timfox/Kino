"""Attack simulators for PEMark (delete/tamper/insert) — Sec. IV-A."""

from __future__ import annotations

import random


def deletion_attack(keys: list[str], *, intensity: float, seed: int = 0) -> list[str]:
    """Randomly delete M keys, M/N = intensity."""
    rng = random.Random(seed)
    keys = keys[:]
    N = len(keys)
    if N == 0:
        return []
    M = int(round(max(0.0, min(1.0, intensity)) * N))
    if M <= 0:
        return keys
    keep = [True] * N
    for idx in rng.sample(range(N), k=min(M, N)):
        keep[idx] = False
    return [k for i, k in enumerate(keys) if keep[i]]


def tamper_attack_values_stub(keys: list[str], *, intensity: float) -> list[str]:
    """Tampering changes values not keys; for position encoding watermark, ordering is unchanged."""
    # This stub just returns the same key order to represent 100% similarity under tamper.
    return keys[:]


def insertion_attack(keys: list[str], *, intensity: float, seed: int = 0) -> list[str]:
    """Insert M fake keys (not touching existing order)."""
    rng = random.Random(seed)
    keys = keys[:]
    N = len(keys)
    M = int(round(max(0.0, min(1.0, intensity)) * N))
    fake = [f"__fake_{rng.randrange(10**9)}__" for _ in range(M)]
    # Append by default (models insertion that doesn't disturb ordering of original keys).
    return keys + fake


def watermark_similarity(a: list[int], b: list[int]) -> float:
    if not a and not b:
        return 100.0
    L = min(len(a), len(b))
    if L == 0:
        return 0.0
    match = sum(1 for i in range(L) if a[i] == b[i])
    return 100.0 * match / L

