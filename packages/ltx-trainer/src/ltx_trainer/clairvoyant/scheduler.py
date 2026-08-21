"""Non-preemptive SJF min-heap dispatcher with starvation guard (Sec. 3.4)."""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.clairvoyant.predictor import predict_plong


@dataclass(order=True)
class _HeapEntry:
    priority: float
    arrival: float
    request_id: str = field(compare=False)
    prompt: str = field(compare=False, default="")


@dataclass
class QueuedRequest:
    request_id: str
    prompt: str
    p_long: float
    arrival_time: float


class SJFScheduler:
    """
    Min-heap keyed on ascending P(Long); starvation promotes longest-waiting request.
    """

    def __init__(self, *, starvation_tau_s: float = 10.5) -> None:
        self.starvation_tau_s = starvation_tau_s
        self._heap: list[_HeapEntry] = []
        self._counter = 0

    def __len__(self) -> int:
        return len(self._heap)

    def enqueue(self, request_id: str, prompt: str, *, p_long: float | None = None, now: float | None = None) -> QueuedRequest:
        ts = time.monotonic() if now is None else now
        pl = predict_plong(prompt) if p_long is None else p_long
        heapq.heappush(self._heap, _HeapEntry(pl, ts, request_id, prompt))
        return QueuedRequest(request_id, prompt, pl, ts)

    def _starvation_candidate(self, now: float) -> _HeapEntry | None:
        if not self._heap:
            return None
        oldest = min(self._heap, key=lambda e: e.arrival)
        if now - oldest.arrival >= self.starvation_tau_s:
            return oldest
        return None

    def pop_next(self, *, now: float | None = None) -> QueuedRequest | None:
        if not self._heap:
            return None
        ts = time.monotonic() if now is None else now
        starved = self._starvation_candidate(ts)
        if starved is not None:
            self._heap.remove(starved)
            heapq.heapify(self._heap)
            return QueuedRequest(starved.request_id, starved.prompt, starved.priority, starved.arrival)
        entry = heapq.heappop(self._heap)
        return QueuedRequest(entry.request_id, entry.prompt, entry.priority, entry.arrival)

    def remove(self, request_id: str) -> bool:
        before = len(self._heap)
        self._heap = [e for e in self._heap if e.request_id != request_id]
        if len(self._heap) != before:
            heapq.heapify(self._heap)
            return True
        return False


def fcfs_dispatch_order(requests: list[tuple[str, str, float]]) -> list[str]:
    """Arrival-order baseline: (id, prompt, arrival_time)."""
    return [r[0] for r in sorted(requests, key=lambda x: x[2])]


def sjf_dispatch_order(
    requests: list[tuple[str, str, float]],
    *,
    starvation_tau_s: float = 10.5,
    variant: str = "sharegpt",
    now_start: float = 0.0,
) -> list[str]:
    """
    Simulate serial dispatch: at each step pick min P(Long) or starved longest-waiting.
    Returns request ids in dispatch order.
    """
    if not requests:
        return []
    sched = SJFScheduler(starvation_tau_s=starvation_tau_s)
    pending = sorted(requests, key=lambda x: x[2])
    order: list[str] = []
    t = now_start
    idx = 0
    in_flight_until = now_start

    while idx < len(pending) or sched:
        while idx < len(pending) and pending[idx][2] <= t:
            rid, prompt, arr = pending[idx]
            pl = predict_plong(prompt, variant=variant)
            sched.enqueue(rid, prompt, p_long=pl, now=arr)
            idx += 1
        if t < in_flight_until:
            t = in_flight_until
            continue
        nxt = sched.pop_next(now=t)
        if nxt is None:
            if idx < len(pending):
                t = pending[idx][2]
            else:
                break
            continue
        order.append(nxt.request_id)
        # placeholder service — caller may extend with service times
        in_flight_until = t + 0.001
        t = in_flight_until
    return order


def validate_short_before_long(
    short_ids: list[str],
    long_ids: list[str],
    dispatch_order: list[str],
) -> dict[str, Any]:
    """End-to-end dispatch test: all Short complete before any Long (Sec. 5.4)."""
    short_set = set(short_ids)
    long_set = set(long_ids)
    last_short_idx = max((dispatch_order.index(s) for s in short_ids if s in dispatch_order), default=-1)
    first_long_idx = min((dispatch_order.index(l) for l in long_ids if l in dispatch_order), default=len(dispatch_order))
    ok = last_short_idx < first_long_idx if short_ids and long_ids else True
    return {
        "sjf_order_valid": ok,
        "last_short_index": last_short_idx,
        "first_long_index": first_long_idx,
        "dispatch_order": dispatch_order,
        "n_short": len(short_set & set(dispatch_order)),
        "n_long": len(long_set & set(dispatch_order)),
    }
