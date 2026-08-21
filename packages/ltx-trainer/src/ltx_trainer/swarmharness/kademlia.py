"""Kademlia-style XOR routing helpers (Sec. 3.2, Maymounkov & Mazières IPTPS 2002)."""

from __future__ import annotations

import hashlib
from typing import Iterable


def key_bytes(skill_name: str) -> bytes:
    return hashlib.sha256(skill_name.encode("utf-8")).digest()


def xor_distance(key_a: bytes, key_b: bytes) -> int:
    """XOR metric on 256-bit keys."""
    if len(key_a) != len(key_b):
        raise ValueError("keys must be same length")
    dist = 0
    for a, b in zip(key_a, key_b, strict=True):
        dist = (dist << 8) | (a ^ b)
    return dist


def closest_nodes(
    target_skill: str,
    node_ids: Iterable[str],
    *,
    k: int = 8,
) -> list[str]:
    """
    Return k node_ids whose skill keys are closest to target_skill under XOR.
    node_id is hashed to a 256-bit routing id for demo purposes.
    """
    target = key_bytes(target_skill)
    scored: list[tuple[int, str]] = []
    for nid in node_ids:
        rid = hashlib.sha256(nid.encode("utf-8")).digest()
        scored.append((xor_distance(target, rid), nid))
    scored.sort(key=lambda x: x[0])
    return [nid for _, nid in scored[:k]]
