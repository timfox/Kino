"""Time-Expanded Network (TEN) representation (Section 4.2, Algorithm 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimeExpandedNetwork:
    """TEN[t][src][dest] = link available at timestep t (spatial adj repeated per t)."""

    adj: dict[int, list[int]]
    max_timesteps: int = 32
    disabled: set[tuple[int, int, int]] = field(default_factory=set)

    @classmethod
    def from_adjacency(cls, adj: dict[int, list[int]], max_timesteps: int = 32) -> TimeExpandedNetwork:
        return cls(adj=adj, max_timesteps=max_timesteps)

    def _link_ok(self, t: int, src: int, dest: int) -> bool:
        if t >= self.max_timesteps:
            return False
        if (t, src, dest) in self.disabled:
            return False
        return dest in self.adj.get(src, [])

    def next_devices(self, npu: int, time: int) -> list[int]:
        return [d for d in self.adj.get(npu, []) if self._link_ok(time, npu, d)]

    def available(self, npu: int, time: int) -> bool:
        return len(self.next_devices(npu, time)) > 0

    def next_available_time(self, npu: int, time: int) -> int:
        t = time
        while t < self.max_timesteps and not self.available(npu, t):
            t += 1
        return t

    def remove_path(self, path: dict[int, list[tuple[int, int]]]) -> None:
        """Remove TEN links used by synthesized paths (Algorithm 3 line 13)."""
        for hops in path.values():
            for t, dest in hops:
                # find current holder from path structure — disable src→dest at t
                self.disabled.add((t, _infer_src(hops, t, dest), dest))

    def snapshot(self) -> dict[str, Any]:
        active = sum(
            1
            for t in range(self.max_timesteps)
            for s, ds in self.adj.items()
            for d in ds
            if self._link_ok(t, s, d)
        )
        return {"max_timesteps": self.max_timesteps, "active_links": active, "disabled": len(self.disabled)}


def _infer_src(hops: list[tuple[int, int]], t: int, dest: int) -> int:
    for i, (ht, hd) in enumerate(hops):
        if ht == t and hd == dest:
            return hops[i - 1][1] if i > 0 else dest
    return dest
