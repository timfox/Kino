"""Host-side forwardRays() logic (Sec. 4.2)."""

from __future__ import annotations

from typing import Generic, TypeVar

from ltx_trainer.rafi.device import RayQueues, sort_rays_by_destination

T = TypeVar("T")


def tally_send_segments(destinations: list[int], num_ranks: int) -> tuple[list[int], list[int]]:
    """send_offset[r], send_count[r] for sorted destination array (Sec. 4.2.2 Step 1)."""
    send_count = [0] * num_ranks
    send_offset = [-1] * num_ranks
    if not destinations:
        return send_count, send_offset

    i = 0
    while i < len(destinations):
        dest = destinations[i]
        j = i + 1
        while j < len(destinations) and destinations[j] == dest:
            j += 1
        if send_offset[dest] < 0:
            send_offset[dest] = i
        send_count[dest] += j - i
        i = j

    for r in range(num_ranks):
        if send_offset[r] < 0:
            send_offset[r] = len(destinations)
    return send_count, send_offset


def simulate_alltoallv(
    sorted_rays: list[T],
    sorted_dests: list[int],
    *,
    num_ranks: int,
    rank: int,
) -> tuple[list[T], int]:
    """Route sorted rays to per-rank buckets (Alltoallv stub)."""
    buckets: list[list[T]] = [[] for _ in range(num_ranks)]
    for ray, dest in zip(sorted_rays, sorted_dests, strict=True):
        buckets[dest].append(ray)
    total = sum(len(b) for b in buckets)
    return buckets[rank], total


def forward_rays_smoke(
    queues: RayQueues[T],
    *,
    num_ranks: int,
    rank: int = 0,
    per_rank_outgoing: list[list[T]] | None = None,
    per_rank_dests: list[list[int]] | None = None,
) -> dict[str, int | list[int]]:
    """Sort, tally, and route work items across ranks (collaborative forwardRays stub)."""
    if per_rank_outgoing is not None and per_rank_dests is not None:
        all_rays: list[T] = []
        all_dests: list[int] = []
        for rays, dests in zip(per_rank_outgoing, per_rank_dests, strict=True):
            all_rays.extend(rays)
            all_dests.extend(dests)
        sorted_rays, sorted_dests = sort_rays_by_destination(all_rays, all_dests)
    else:
        sorted_rays, sorted_dests = sort_rays_by_destination(queues.outgoing, queues.destinations)

    send_count, send_offset = tally_send_segments(sorted_dests, num_ranks)
    received, total_system = simulate_alltoallv(
        sorted_rays,
        sorted_dests,
        num_ranks=num_ranks,
        rank=rank,
    )

    queues.incoming = received
    queues.outgoing = []
    queues.destinations = []

    return {
        "num_emitted": len(sorted_rays),
        "num_received": len(received),
        "total_rays_in_system": total_system,
        "send_count": send_count,
        "send_offset": send_offset,
    }
