"""Vector-level update mechanism (Section 3)."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from ltx_trainer.yi.graph import ProximityGraph, beam_search, prune_neighbors


@dataclass
class UpdateStats:
    connect_tasks: int = 0
    vectors_visited: int = 0
    inserts: int = 0
    deletes: int = 0
    sampled_connects: int = 0


@dataclass
class VectorLevelUpdater:
    """Unify Insert/Delete expand+prune into per-vector connect tasks."""

    graph: ProximityGraph
    delete_lru_capacity: int = 8
    delete_list: OrderedDict[int, None] = field(default_factory=OrderedDict)
    connect_list: list[int] = field(default_factory=list)
    stats: UpdateStats = field(default_factory=UpdateStats)

    def _touch_delete(self, vid: int) -> None:
        if vid in self.delete_list:
            self.delete_list.move_to_end(vid)
        else:
            self.delete_list[vid] = None
        while len(self.delete_list) > self.delete_lru_capacity:
            self.delete_list.popitem(last=False)

    def insert(self, vid: int, raw: tuple[float, ...] | None = None) -> None:
        nbrs = beam_search(self.graph, raw or (float(vid),), k=min(3, self.graph.max_degree))
        self.graph.add(vid, neighbors=[], raw=raw)
        self.connect_list.extend(nbrs)
        self.stats.inserts += 1
        self._flush_connects(inserted=vid)

    def delete(self, vid: int) -> None:
        self.graph.mark_deleted(vid)
        self._touch_delete(vid)
        self.stats.deletes += 1
        # Delete-heavy: sample consecutive ids to feed connect list
        if not self.connect_list:
            alive = self.graph.alive()
            sample = alive[: min(3, len(alive))]
            self.connect_list.extend(sample)
            self.stats.sampled_connects += len(sample)
        self._flush_connects(inserted=None)

    def _flush_connects(self, inserted: int | None) -> None:
        pending = list(dict.fromkeys(self.connect_list))  # unique preserve order
        self.connect_list.clear()
        for u in pending:
            if u not in self.graph.vertices or self.graph.vertices[u].deleted:
                continue
            self._connect_task(u, inserted=inserted)

    def _connect_task(self, u: int, inserted: int | None) -> None:
        node = self.graph.vertices[u]
        expanded = list(node.neighbors)
        self.stats.vectors_visited += 1
        if inserted is not None and inserted not in expanded:
            expanded.append(inserted)
        # Expand for deletes: drop deleted, append their out-neighbors
        new_list: list[int] = []
        for nb in expanded:
            if nb in self.delete_list:
                deleted_node = self.graph.vertices.get(nb)
                if deleted_node:
                    new_list.extend(deleted_node.neighbors)
                self.stats.vectors_visited += 1
            else:
                new_list.append(nb)
        node.neighbors = prune_neighbors(self.graph, u, new_list)
        self.stats.connect_tasks += 1
