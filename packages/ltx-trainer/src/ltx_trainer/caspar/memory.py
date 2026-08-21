"""Symbolic memory accessors and blocked struct-of-arrays layout (§IV)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np

Array = np.ndarray


class AccessPattern(str, Enum):
    SEQUENTIAL_READ = "sequential_read"
    SEQUENTIAL_WRITE = "sequential_write"
    INDEXED_READ = "indexed_read"
    INDEXED_WRITE = "indexed_write"
    INDEXED_ADD = "indexed_add"
    UNIQUE_READ = "unique_read"
    UNIQUE_ADD = "unique_add"
    SUM_WRITE = "sum_write"
    SUM_ADD = "sum_add"
    PAIRWISE_READ = "pairwise_read"
    PAIRWISE_WRITE = "pairwise_write"
    SHARED_READ = "shared_read"
    SHARED_ADD = "shared_add"


@dataclass
class SharedIndex:
    """Paper §IV-A SharedIndex struct for warp-efficient sparse access."""

    unique: int
    target: int
    argsort: int


def build_shared_index(indices: Array) -> list[SharedIndex]:
    """Build sorted block-unique indexing structs from per-thread indices."""
    uniq, inverse = np.unique(indices, return_inverse=True)
    order = np.argsort(indices, kind="stable")
    argsort = np.empty_like(order)
    argsort[order] = np.arange(len(order))
    out: list[SharedIndex] = []
    for i, u in enumerate(indices):
        out.append(SharedIndex(unique=int(uniq[inverse[i]]), target=int(inverse[i]), argsort=int(argsort[i])))
    return out


def sequential_read(data: Array, thread_idx: Array) -> Array:
    return data[thread_idx]


def indexed_add(data: Array, indices: Array, values: Array) -> Array:
    out = data.copy()
    np.add.at(out, indices, values)
    return out


def blocked_struct_of_arrays(
    records: Array,
    *,
    chunk: int = 4,
) -> tuple[Array, Array]:
    """
    Map array-of-structs (N, F) to blocked SoA chunks (§IV-B, Fig. 2).

    Returns (blocked_data, chunk_map) where blocked_data holds ceil(F/chunk) blocks of size chunk.
    """
    n, f = records.shape
    pad_f = ((f + chunk - 1) // chunk) * chunk
    padded = np.zeros((n, pad_f), dtype=records.dtype)
    padded[:, :f] = records
    n_chunks = pad_f // chunk
    blocked = padded.reshape(n, n_chunks, chunk).transpose(1, 0, 2).reshape(n_chunks * n, chunk)
    chunk_map = np.arange(n_chunks)
    return blocked, chunk_map


def array_of_structs_from_blocked(blocked: Array, *, n: int, f: int, chunk: int = 4) -> Array:
    n_chunks = ((f + chunk - 1) // chunk)
    pad_f = n_chunks * chunk
    rev = blocked.reshape(n_chunks, n, chunk).transpose(1, 0, 2).reshape(n, pad_f)
    return rev[:, :f]


def accessor_dispatch(pattern: AccessPattern, data: Array, indices: Array, values: Array | None = None) -> Array:
    if pattern is AccessPattern.SEQUENTIAL_READ:
        return sequential_read(data, indices)
    if pattern is AccessPattern.INDEXED_ADD and values is not None:
        return indexed_add(data, indices, values)
    if pattern is AccessPattern.SHARED_READ:
        structs = build_shared_index(indices)
        gather = np.array([data[s.unique] for s in structs])
        return gather
    raise NotImplementedError(f"accessor stub: {pattern}")
