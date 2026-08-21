"""Tasklet engine, buffer manager, and vector file system stubs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.yi.config import YiConfig


@dataclass
class TaskletEngine:
    """Coroutine-style search/connect queues (CPU simulation)."""

    search_queue: list[str] = field(default_factory=list)
    connect_queue: list[str] = field(default_factory=list)
    suspended_io: int = 0
    resumed: int = 0

    def schedule_insert(self, vid: int, n_neighbors: int = 3) -> None:
        self.search_queue.append(f"search:{vid}")
        for i in range(n_neighbors):
            self.connect_queue.append(f"connect:{vid}:{i}")

    def drain(self, overlap_io: bool = True) -> dict[str, int]:
        processed = 0
        while self.search_queue or self.connect_queue:
            if self.search_queue:
                self.search_queue.pop(0)
                if overlap_io:
                    self.suspended_io += 1
                    self.resumed += 1
                processed += 1
            if self.connect_queue:
                self.connect_queue.pop(0)
                processed += 1
        return {
            "processed": processed,
            "suspended_io": self.suspended_io,
            "resumed": self.resumed,
        }


@dataclass
class AsyncBufferManager:
    """Fixed-budget page cache with dirty/free lists."""

    capacity: int = 8
    buffers: dict[int, dict[str, Any]] = field(default_factory=dict)
    free_list: list[int] = field(default_factory=list)
    dirty_list: list[int] = field(default_factory=list)
    hits: int = 0
    misses: int = 0

    def access(self, page_id: int, *, write: bool = False) -> None:
        if page_id in self.buffers:
            self.hits += 1
            buf = self.buffers[page_id]
            buf["ref"] = int(buf.get("ref", 0)) + 1
            if write:
                buf["dirty"] = True
                if page_id not in self.dirty_list:
                    self.dirty_list.append(page_id)
            buf["ref"] = max(0, int(buf["ref"]) - 1)
            return
        self.misses += 1
        if len(self.buffers) >= self.capacity and self.free_list:
            evict = self.free_list.pop(0)
            self.buffers.pop(evict, None)
        elif len(self.buffers) >= self.capacity:
            # Evict any zero-ref non-dirty
            for pid, buf in list(self.buffers.items()):
                if int(buf.get("ref", 0)) == 0 and not buf.get("dirty"):
                    del self.buffers[pid]
                    break
        self.buffers[page_id] = {"ref": 0, "dirty": write}
        if write and page_id not in self.dirty_list:
            self.dirty_list.append(page_id)
        if page_id not in self.free_list and not write:
            self.free_list.append(page_id)

    def flush_dirty(self) -> int:
        n = len(self.dirty_list)
        for pid in self.dirty_list:
            if pid in self.buffers:
                self.buffers[pid]["dirty"] = False
                if pid not in self.free_list:
                    self.free_list.append(pid)
        self.dirty_list.clear()
        return n

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


@dataclass
class VectorFileSystem:
    """Split navigation (index) vs raw data blocks."""

    split_layout: bool = True
    index_block_accesses: int = 0
    data_block_accesses: int = 0

    def access_for_update(self, vid: int) -> None:
        self.index_block_accesses += 1
        # split layout: updates skip raw vector data blocks
        if not self.split_layout:
            self.data_block_accesses += 1

    def access_for_search(self, vid: int, *, need_raw: bool = True) -> None:
        self.index_block_accesses += 1
        if need_raw:
            self.data_block_accesses += 1

    def cache_friendliness(self) -> float:
        """Higher when updates avoid data blocks (paper +Layout)."""
        total = self.index_block_accesses + self.data_block_accesses
        if total == 0:
            return 0.0
        return self.index_block_accesses / total


def component_breakdown(config: YiConfig | None = None) -> dict[str, Any]:
    """Figure 17 style cumulative speedups (paper anchors as relative factors)."""
    cfg = config or YiConfig()
    # Paper: +Tasklet 4.02×, +Buffer 1.33×, +Layout 1.04× (sequential on baseline)
    factors = {
        "baseline": 1.0,
        "tasklet": 4.02 if cfg.enable_tasklets else 1.0,
        "buffer": 1.33 if cfg.enable_buffer_persist else 1.0,
        "layout": 1.04 if cfg.enable_split_layout else 1.0,
    }
    cumulative = 1.0
    steps = []
    for name in ("baseline", "tasklet", "buffer", "layout"):
        if name != "baseline":
            cumulative *= factors[name]
        steps.append({"stage": f"+{name}" if name != "baseline" else "Baseline", "rel_throughput": round(cumulative, 3)})
    return {"factors": factors, "steps": steps, "final_rel": round(cumulative, 3)}
