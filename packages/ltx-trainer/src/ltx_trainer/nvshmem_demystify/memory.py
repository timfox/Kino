"""Symmetric heap and remote address computation (§IV, Eq. 1)."""

from __future__ import annotations

from typing import Any


def remote_address_p2p(
    dest_local: int,
    heap_base: int,
    peer_heap_base_p2p: int,
) -> int:
    """P2P fast path: dest_remote = peer_base + (dest_local - heap_base)."""
    return peer_heap_base_p2p + (dest_local - heap_base)


def heap_growth_steps() -> list[dict[str, str]]:
    """Steps 1–5 when allocate_physical_memory_to_heap() grows the heap."""
    return [
        {"step": 1, "op": "cuMemCreate()", "detail": "physical memory handle"},
        {"step": 2, "op": "cuMemMap() + cuMemSetAccess()", "detail": "map subrange, peer access"},
        {"step": 3, "op": "P2P transport register", "detail": "intra-node fast path"},
        {"step": 4, "op": "mspace insert", "detail": "host-side allocator chunk"},
        {"step": 5, "op": "network transport register", "detail": "non-P2P peers"},
    ]


def symmetric_heap_card() -> dict[str, Any]:
    return {
        "backend": "CUDA VMM (cuMemAddressReserve + on-demand cuMemCreate)",
        "va_reservation": "p2p_npes_ × heap_size_ per PE",
        "fast_path_table": "peer_heap_base_p2p_[]",
        "slow_path_table": "peer_heap_base_remote_[]",
        "allocator": "host-side mspace first-fit, O(n) scan",
        "remote_address_rule": "offset-preserving symmetric addressing",
    }
