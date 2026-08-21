"""Asynchronous offload/reload of weight pages."""

from __future__ import annotations

from dataclasses import dataclass, field

from ltx_trainer.pagedweight.pages import WeightPageTable
from ltx_trainer.pagedweight.planner import PlanResult


@dataclass
class MovementStats:
    offloads: int = 0
    reloads: int = 0
    bytes_offloaded: int = 0
    bytes_reloaded: int = 0
    pending_transfers: int = 0


@dataclass
class AsyncPageMover:
    """Commit lower/higher bitwidths only at safe boundaries (Figure 4)."""

    stats: MovementStats = field(default_factory=MovementStats)
    reload_threshold_free_blocks: int = 8

    def execute_offload(self, table: WeightPageTable, plan: PlanResult) -> int:
        """Offload: lower qi then move unneeded pages GPU→CPU."""
        n = 0
        for a in plan.actions:
            page = table.pages[a.key]
            if page.desired_bits >= page.committed_bits:
                continue
            page.page_state = "transfer"
            released = page.released_bytes(page.committed_bits, page.desired_bits)
            page.committed_bits = page.desired_bits
            page.page_state = "gpu"  # remaining bit-planes stay GPU-resident
            self.stats.offloads += 1
            self.stats.bytes_offloaded += released
            n += 1
        return n

    def maybe_reload(self, table: WeightPageTable, free_blocks: int, restore_bits: int = 8) -> int:
        """Reload when KV pressure eases: restore pages then raise qi."""
        if free_blocks < self.reload_threshold_free_blocks:
            return 0
        n = 0
        for page in table.pages.values():
            if page.committed_bits >= restore_bits:
                continue
            page.page_state = "transfer"
            gained = page.released_bytes(restore_bits, page.committed_bits)
            page.committed_bits = restore_bits
            page.desired_bits = restore_bits
            page.page_state = "gpu"
            self.stats.reloads += 1
            self.stats.bytes_reloaded += gained
            n += 1
        return n

    def as_dict(self) -> dict:
        return {
            "offloads": self.stats.offloads,
            "reloads": self.stats.reloads,
            "bytes_offloaded": self.stats.bytes_offloaded,
            "bytes_reloaded": self.stats.bytes_reloaded,
        }
