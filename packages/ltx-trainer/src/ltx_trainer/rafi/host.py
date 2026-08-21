"""RaFI host context (Sec. 3.4)."""

from __future__ import annotations

from typing import Generic, TypeVar

from ltx_trainer.rafi.device import DeviceInterface, RayQueues
from ltx_trainer.rafi.forward import forward_rays_smoke

T = TypeVar("T")


class HostContext(Generic[T]):
    """resizeRayQueues / getDeviceInterface / forwardRays."""

    def __init__(self, *, rank: int = 0, num_ranks: int = 1, max_capacity: int = 4096) -> None:
        self.rank = rank
        self.num_ranks = num_ranks
        self._max_capacity = max_capacity
        self._queues = RayQueues[T](incoming=[], outgoing=[], destinations=[], max_capacity=max_capacity)

    def resize_ray_queues(self, n: int) -> None:
        self._max_capacity = int(n)
        self._queues.max_capacity = int(n)

    def get_device_interface(self) -> DeviceInterface[T]:
        return DeviceInterface(self._queues)

    def forward_rays(
        self,
        *,
        per_rank_outgoing: list[list[T]] | None = None,
        per_rank_dests: list[list[int]] | None = None,
    ) -> int:
        stats = forward_rays_smoke(
            self._queues,
            num_ranks=self.num_ranks,
            rank=self.rank,
            per_rank_outgoing=per_rank_outgoing,
            per_rank_dests=per_rank_dests,
        )
        return int(stats["total_rays_in_system"])
