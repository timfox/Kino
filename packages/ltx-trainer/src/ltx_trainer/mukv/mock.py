"""MuKV compression smoke."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def toy_keys(n: int, dim: int, *, phase: float = 0.0) -> list[list[float]]:
    rows: list[list[float]] = []
    for i in range(n):
        rows.append(
            [math.sin(2.0 * math.pi * ((i * dim + j) / max(n * dim, 1) + phase)) for j in range(dim)]
        )
    return rows


def toy_attention(*, num_heads: int = 4, num_tokens: int = 16) -> list[list[float]]:
    rng = np.random.default_rng(0)
    rows: list[list[float]] = []
    for _ in range(num_heads):
        row = rng.random(num_tokens)
        s = float(row.sum())
        rows.append([float(x / s) for x in row])
    return rows


def toy_question_vec(*, dim: int = 8) -> list[float]:
    rng = np.random.default_rng(1)
    v = rng.standard_normal(dim)
    n = float(np.linalg.norm(v)) + 1e-9
    return [float(x / n) for x in v]


def evaluation_smoke() -> dict[str, Any]:
    from ltx_trainer.mukv.compression import attention_importance, compress_indices

    keys = toy_keys(16, dim=8)
    attn = toy_attention(num_heads=4, num_tokens=16)
    imp = attention_importance(attn)
    kept = compress_indices(imp, rho=0.5)
    return {"num_keys": len(keys), "importance_len": len(imp), "kept_indices": len(kept)}
