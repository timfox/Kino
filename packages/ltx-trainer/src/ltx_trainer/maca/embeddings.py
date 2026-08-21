"""Tiny deterministic embeddings for MACA reference demos.

The paper uses a sentence encoder for task + agent embeddings. In this repo we avoid heavyweight
dependencies and provide a deterministic hashing encoder suitable for unit tests and toy demos.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

import torch
from torch import Tensor

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _hash_u32(text: str) -> int:
    h = hashlib.blake2s(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, "little", signed=False)


def embed_text(text: str, *, dim: int, seed: int = 0) -> Tensor:
    """Return a unit-norm vector embedding of ``text`` with deterministic hashing."""
    t = (text or "").lower()
    toks = _TOKEN_RE.findall(t)
    if not toks:
        toks = ["<empty>"]
    v = torch.zeros(dim, dtype=torch.float32)
    for tok in toks:
        idx = (_hash_u32(f"{seed}:{tok}") % dim)
        sign = -1.0 if (_hash_u32(f"s:{seed}:{tok}") & 1) else 1.0
        v[idx] += sign
    n = float(torch.linalg.vector_norm(v).item())
    if n <= 1e-12:
        v[0] = 1.0
        n = 1.0
    return v / n


def cosine(a: Tensor, b: Tensor) -> float:
    a = a / (torch.linalg.vector_norm(a) + 1e-12)
    b = b / (torch.linalg.vector_norm(b) + 1e-12)
    return float(torch.dot(a, b).item())


def budget_temperature(budget_tokens: int, *, base: float) -> float:
    """Budget-dependent temperature β(b): higher budget → lower temperature (sharper)."""
    b = max(1, int(budget_tokens))
    return float(base * (1.0 + 400.0 / math.sqrt(b)))


@dataclass(frozen=True)
class AgentProfile:
    name: str
    capability: str
    expected_cost_tokens: int

    def embed(self, *, dim: int) -> Tensor:
        return embed_text(f"{self.name} {self.capability}", dim=dim, seed=17)

