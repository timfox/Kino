"""RaFI device interface (Sec. 3.3) — host/device queue helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from torch import Tensor

T = TypeVar("T")


@dataclass
class RayQueues(Generic[T]):
    """Separate input/output arrays + destination ranks (Sec. 3.2)."""

    incoming: list[T]
    outgoing: list[T]
    destinations: list[int]
    max_capacity: int

    def __post_init__(self) -> None:
        if len(self.destinations) != len(self.outgoing):
            raise ValueError("destinations must align with outgoing queue")


class DeviceInterface(Generic[T]):
    """Trivially copyable interface passed to CUDA kernels (paper Sec. 3.3)."""

    def __init__(self, queues: RayQueues[T]) -> None:
        self._queues = queues

    def num_incoming(self) -> int:
        return len(self._queues.incoming)

    def get_incoming(self, ray_id: int) -> T:
        return self._queues.incoming[ray_id]

    def emit_outgoing(self, ray: T, dest: int) -> bool:
        """Append ray to output queue; return False on overflow."""
        if len(self._queues.outgoing) >= self._queues.max_capacity:
            return False
        self._queues.outgoing.append(ray)
        self._queues.destinations.append(int(dest))
        return True


def pack_sort_keys(destinations: list[int], indices: list[int] | None = None) -> Tensor:
    """uint64 keys: upper 32 = dest rank, lower 32 = source index (Sec. 4.2.1)."""
    n = len(destinations)
    if indices is None:
        indices = list(range(n))
    keys = []
    for dest, src in zip(destinations, indices, strict=True):
        keys.append((int(dest) << 32) | (int(src) & 0xFFFFFFFF))
    return torch.tensor(keys, dtype=torch.int64)


def unpack_sort_key(key: int) -> tuple[int, int]:
    dest = (key >> 32) & 0xFFFFFFFF
    src = key & 0xFFFFFFFF
    return int(dest), int(src)


def sort_rays_by_destination(
    rays: list[T],
    destinations: list[int],
) -> tuple[list[T], list[int]]:
    """CPU radix-style sort by destination rank (mirrors CUB step)."""
    if len(rays) != len(destinations):
        raise ValueError("rays and destinations length mismatch")
    order = sorted(range(len(destinations)), key=lambda i: destinations[i])
    return [rays[i] for i in order], [destinations[i] for i in order]
