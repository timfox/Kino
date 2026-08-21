"""Pre-commit Merkle roots and per-step anchor chain (MOD. 2–4)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


def merkle_root(values: np.ndarray, *, leaf_bytes: int = 4096) -> bytes:
    """Toy Merkle apex over flattened tensor bytes."""
    flat = values.astype(np.float32).tobytes()
    chunks = [flat[i : i + leaf_bytes] for i in range(0, max(len(flat), 1), leaf_bytes)] or [b""]
    layer = [hashlib.sha256(c).digest() for c in chunks]
    while len(layer) > 1:
        nxt: list[bytes] = []
        for i in range(0, len(layer), 2):
            pair = layer[i : i + 2]
            if len(pair) == 1:
                pair.append(pair[0])
            nxt.append(hashlib.sha256(pair[0] + pair[1]).digest())
        layer = nxt
    return layer[0]


def compound_commitment(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()


@dataclass
class PreCommitment:
    arch_spec_hash: bytes
    dataset_root: bytes
    weights_root: bytes
    ex_ante_hash: bytes | None = None

    def h_commit(self) -> bytes:
        parts = [self.arch_spec_hash, self.dataset_root, self.weights_root]
        if self.ex_ante_hash:
            parts.append(self.ex_ante_hash)
        return compound_commitment(*parts)


@dataclass
class AnchorChain:
    anchor_init: bytes = field(default=b"")
    links: list[tuple[int, bytes, bytes]] = field(default_factory=list)

    def link(self, step: int, weight_root: bytes, tap_tag: bytes) -> None:
        prev = self.anchor_init if not self.links else self.links[-1][1]
        payload = hashlib.sha256(
            b"ANCHOR/LINK" + prev + weight_root + tap_tag + step.to_bytes(8, "little")
        ).digest()
        self.links.append((step, payload, tap_tag))

    def terminal(self) -> bytes:
        return self.links[-1][1] if self.links else self.anchor_init
